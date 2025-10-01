#!/usr/bin/env python3
"""
MindCanvas Backend - YOLOv5 HTP 이미지 분석 API
Flask를 사용한 웹 인터페이스
"""

import os
import sys
import json
import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin # cross_origin 임포트 추가
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import yolov5
import yolov5.models # Add this line to import models
from htp_analyzer import HTPAnalyzer
from dotenv import load_dotenv
import openai
import httpx
import json
import uuid # 그림 파일명 생성을 위해 추가
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import traceback # traceback 모듈 임포트
import jwt
from datetime import datetime, timedelta
from functools import wraps

# 환경변수 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 네이버 API 키 설정
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')
NAVER_SEARCH_CLIENT_ID = os.getenv('NAVER_SEARCH_CLIENT_ID')
NAVER_SEARCH_CLIENT_SECRET = os.getenv('NAVER_SEARCH_CLIENT_SECRET')

app = Flask(__name__)

CORS(app)

# PostgreSQL 설정
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://your_user:your_password@localhost:5432/your_database') + '?client_encoding=UTF8'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 기존 설정 다시 추가
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-this-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)  # 30일간 유효
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 사용자 모델 정의
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    def __repr__(self):
        return f'<User {self.username}>'

# JWT 토큰 관련 함수들
def generate_jwt_token(user_id, username):
    """JWT 토큰 생성"""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES'],
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    return token

def verify_jwt_token(token):
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """토큰 검증 데코레이터"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # "Bearer <token>" 형식에서 토큰 추출
            except IndexError:
                return jsonify({"error": "토큰 형식이 올바르지 않습니다."}), 401
        
        if not token:
            return jsonify({"error": "토큰이 필요합니다."}), 401
        
        try:
            payload = verify_jwt_token(token)
            if payload is None:
                return jsonify({"error": "유효하지 않은 토큰입니다."}), 401
            
            # request 객체에 사용자 정보 추가
            request.current_user = payload
        except Exception as e:
            return jsonify({"error": f"토큰 검증 오류: {str(e)}"}), 401
        
        return f(*args, **kwargs)
    return decorated

# 그림 모델 정의
class Drawing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(256), nullable=False)
    analysis_result = db.Column(db.JSON, nullable=True) # JSON 타입으로 분석 결과 저장
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('drawings', lazy=True))

    def __repr__(self):
        return f'<Drawing {self.id} by User {self.user_id}>'

# 업로드 및 출력 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class YOLOv5HTPAnalyzer:
    def __init__(self):
        self.device = 'cpu'  # 웹에서는 CPU 사용
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """모든 YOLOv5 HTP 모델 로드"""
        # PyTorch 2.8.0+에서 모델 로딩 문제 해결
        try:
            # YOLOv5 모델 클래스를 안전한 글로벌로 등록
            torch.serialization.add_safe_globals([yolov5.models.yolo.Model])
            print("✅ PyTorch 안전 글로벌 설정 완료")
        except Exception as e:
            print(f"PyTorch 안전 글로벌 설정 경고: {e}")
       
        # torch.load를 래핑하여 weights_only=False 설정
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        
        model_configs = {
            "House": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/House/exp/weights/best.pt",
                "classes": ["집", "지붕", "문", "창문", "굴뚝", "연기", "울타리", "길", "연못", "산", "나무", "꽃", "잔디", "태양"]
            },
            "PersonF": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/PersonF/exp/weights/best.pt",
                "classes": ["머리", "얼굴", "눈", "코", "입", "귀", "머리카락", "목", "상체", "팔", "손", "다리", "발", "단추", "주머니", "운동화", "여자구두"]
            },
            "PersonM": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/PersonM/exp/weights/best.pt",
                "classes": ["머리", "얼굴", "눈", "코", "입", "귀", "머리카락", "목", "상체", "팔", "손", "다리", "발", "단추", "주머니", "운동화", "남자구두"]
            },
            "Tree": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/Tree/exp/weights/best.pt",
                "classes": ["나무", "기둥", "수관", "가지", "뿌리", "나뭇잎", "꽃", "열매", "그네", "새", "다람쥐", "구름", "달", "별"]
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                if os.path.exists(config["weights"]):
                    model = yolov5.load(config["weights"])
                    model.conf = 0.25  # 기본 신뢰도 임계값
                    model.iou = 0.45   # 기본 IoU 임계값
                    self.models[model_name] = {
                        "model": model,
                        "classes": config["classes"]
                    }
                    print(f"✅ {model_name} 모델 로드 완료")
                else:
                    print(f"❌ {model_name} 모델 파일을 찾을 수 없습니다: {config['weights']}")
            except Exception as e:
                print(f"❌ {model_name} 모델 로드 실패: {e}")
    
    def predict(self, image, model_name, conf_threshold=0.25, iou_threshold=0.45):
        """이미지에 대한 객체 탐지 수행"""
        if model_name not in self.models:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")
        
        model_info = self.models[model_name]
        model = model_info["model"]
        classes = model_info["classes"]
        
        # 모델 설정 업데이트
        model.conf = conf_threshold
        model.iou = iou_threshold
        
        # 예측 수행
        results = model(image)
        
        # 결과 파싱
        detections = []
        if len(results.pred[0]) > 0:
            for *box, conf, cls in results.pred[0]:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                confidence = float(conf)
                
                if class_id < len(classes):
                    detections.append({
                        "class": classes[class_id],
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        return detections
    
    
    

# 전역 분석기 인스턴스
yolo_analyzer = YOLOv5HTPAnalyzer()
htp_analyzer = HTPAnalyzer()

# HTP 해석 기준 로드
def load_interpretation_rules():
    """이미지 분석 해석기준 JSON 파일을 로드합니다."""
    try:
        with open('interpretation/img_int.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("해석기준 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None

# HTP 해석 기준 로드
interpretation_rules = load_interpretation_rules()

def get_htp_system_prompt():
    """img_int.json의 내용을 기반으로 시스템 프롬프트를 생성합니다."""
    if not interpretation_rules:
        return "HTP 해석기준을 로드할 수 없습니다."
    
    instructions = interpretation_rules.get("instructions", [])
    htp_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    examples = interpretation_rules.get("examples", [])
    
    prompt = "당신은 HTP(House-Tree-Person) 그림 검사 해석 전문가입니다.\n\n"
    
    # 시스템 지시사항 추가
    for instruction in instructions:
        if instruction.get("role") == "system":
            prompt += instruction.get("content", "") + "\n\n"
    
    # HTP 해석 기준 추가
    prompt += "HTP 해석 기준:\n"
    for object_type, criteria in htp_criteria.items():
        if object_type == "house":
            prompt += "🏠 집 (House):\n"
        elif object_type == "tree":
            prompt += "🌳 나무 (Tree):\n"
        elif object_type == "person":
            prompt += "👤 사람 (Person):\n"
        
        for feature, description in criteria.items():
            prompt += f"- {feature}: {description}\n"
        prompt += "\n"
    
    # 예시 추가
    if examples:
        prompt += "예시 대화:\n"
        for example in examples[:3]:  # 처음 3개 예시만
            prompt += f"사용자: {example.get('user', '')}\n"
            prompt += f"상담사: {example.get('assistant', '')}\n\n"
    
    prompt += """당신의 역할:
1. 이미지 분석 결과를 받으면 HTP 기준에 따라 심리적 해석을 제공
2. 각 특징별로 점수를 계산하고 위험도를 평가
3. 구체적이고 실용적인 상담 조언 제공
4. 미술심리상담과 그림 해석 관련 질문만 답변"""
    
    return prompt

def analyze_image_features(image_analysis_result):
    """이미지 분석 결과를 HTP 해석기준에 따라 분석합니다."""
    if not interpretation_rules:
        return {"error": "해석기준을 로드할 수 없습니다."}
    
    analysis_result = {
        "objects": {},
        "total_score": 0,
        "interpretations": [],
        "risk_level": "normal"
    }
    
    htp_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    
    # 각 객체별 분석 (집, 나무, 사람)
    for object_type in ["house", "tree", "person"]:
        if object_type not in image_analysis_result:
            continue
            
        object_features = image_analysis_result[object_type]
        object_criteria = htp_criteria.get(object_type, {})
        
        object_analysis = {
            "label": "집" if object_type == "house" else "나무" if object_type == "tree" else "사람",
            "features": {},
            "score": 0,
            "interpretations": []
        }
        
        # 각 특징별 분석
        for feature_name, feature_value in object_features.items():
            # 특징에 따른 해석 생성
            interpretation = generate_interpretation(object_type, feature_name, feature_value, "")
            if interpretation:
                object_analysis["interpretations"].append(interpretation)
                object_analysis["score"] += interpretation.get("score", 0)
                analysis_result["interpretations"].append(interpretation)
        
        analysis_result["objects"][object_type] = object_analysis
        analysis_result["total_score"] += object_analysis["score"]
    
    # 위험도 평가
    if analysis_result["total_score"] <= -5:
        analysis_result["risk_level"] = "high"
    elif analysis_result["total_score"] <= -1:
        analysis_result["risk_level"] = "moderate"
    elif analysis_result["total_score"] >= 4:
        analysis_result["risk_level"] = "positive"
    
    return analysis_result

def generate_interpretation(object_type, feature_name, feature_value, criteria_text):
    """특징값을 기반으로 상세한 해석을 생성합니다."""
    if not interpretation_rules:
        return None
    
    detailed_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    object_criteria = detailed_criteria.get(object_type, {})
    
    # 기본 해석 구조
    interpretation = {
        "feature": feature_name,
        "interpretation": "",
        "severity": "info",
        "score": 0,
        "reasoning": "",
        "threshold": "",
        "psychological_meaning": ""
    }
    
    # 크기 분석
    if feature_name == "size" and isinstance(feature_value, (int, float)):
        size_criteria = object_criteria.get("size", {})
        
        if feature_value >= size_criteria.get("very_large", {}).get("threshold", 0.8):
            criteria = size_criteria["very_large"]
            threshold = size_criteria.get("very_large", {}).get("threshold", 0.8)
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 임계값 {threshold} 이상으로 매우 큼",
                "threshold": f"임계값: {threshold} 이상",
                "psychological_meaning": "HTP 기준에 따르면 화지를 꽉 채우거나 밖으로 벗어날 정도의 큰 크기는 충동적이고 공격적인 성향을 나타냅니다. 이는 자아 통제력 부족이나 과도한 자기 표현 욕구를 의미할 수 있습니다."
            })
        elif feature_value <= size_criteria.get("small", {}).get("threshold", 0.25):
            criteria = size_criteria["small"]
            threshold = size_criteria.get("small", {}).get("threshold", 0.25)
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 임계값 {threshold} 이하로 매우 작음",
                "threshold": f"임계값: {threshold} 이하",
                "psychological_meaning": "HTP 기준에 따르면 1/4 이하의 작은 크기는 대인관계에서의 무력감, 열등감, 불안, 우울적 경향을 나타냅니다. 이는 자신감 부족이나 위축된 자아상을 반영할 수 있습니다."
            })
        else:
            criteria = size_criteria.get("normal", {})
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 정상 범위 내에 있음",
                "threshold": f"정상 범위: 0.25 < 크기 < 0.8",
                "psychological_meaning": "적절한 크기는 균형 잡힌 자아상과 현실적 인식을 나타냅니다."
            })
    
    # 위치 분석
    elif feature_name == "location" and isinstance(feature_value, (int, float)):
        position_criteria = object_criteria.get("position", {})
        
        if feature_value < 0.3:  # 상단
            if "top_view" in position_criteria:
                criteria = position_criteria["top_view"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.3 미만으로 상단에 위치",
                    "threshold": "위치 < 0.3 (상단)",
                    "psychological_meaning": "HTP 기준에 따르면 상단에 위치한 객체는 이상화 성향이나 현실 도피 경향을 나타냅니다. 이는 현실보다 이상적인 세계를 추구하는 심리를 의미할 수 있습니다."
                })
        elif feature_value > 0.7:  # 하단
            if "bottom_half" in position_criteria:
                criteria = position_criteria["bottom_half"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.7 초과로 하단에 위치",
                    "threshold": "위치 > 0.7 (하단)",
                    "psychological_meaning": "HTP 기준에 따르면 하단에 위치한 객체는 불안정감, 우울적 경향을 나타냅니다. 이는 기반 부족이나 불안정한 정서 상태를 의미할 수 있습니다."
                })
        elif feature_value < 0.5:  # 좌측
            if "left" in position_criteria:
                criteria = position_criteria["left"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.5 미만으로 좌측에 위치",
                    "threshold": "위치 < 0.5 (좌측)",
                    "psychological_meaning": "HTP 기준에 따르면 좌측에 위치한 객체는 내향적, 열등감을 나타냅니다. 이는 과거 지향적이거나 소극적인 성향을 의미할 수 있습니다."
                })
        else:  # 우측
            if "right" in position_criteria:
                criteria = position_criteria["right"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.5 이상으로 우측에 위치",
                    "threshold": "위치 >= 0.5 (우측)",
                    "psychological_meaning": "HTP 기준에 따르면 우측에 위치한 객체는 외향성, 활동성을 나타냅니다. 이는 미래 지향적이거나 적극적인 성향을 의미할 수 있습니다."
                })
    
    # 창문 분석
    elif feature_name == "window":
        window_criteria = object_criteria.get("window", {})
        
        if feature_value == 0:
            if "missing" in window_criteria:
                criteria = window_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"창문 개수 {feature_value}개로 창문이 완전히 없음",
                    "threshold": "창문 0개",
                    "psychological_meaning": "HTP 기준 H23에 따르면 창문이 생략된 집은 폐쇄적 사고와 환경에 대한 관심 결여, 적의를 나타냅니다. 이는 사회적 교류 회피나 외부 세계에 대한 방어적 태도를 의미합니다."
                })
        elif feature_value >= 3:
            if "many" in window_criteria:
                criteria = window_criteria["many"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"창문 개수 {feature_value}개로 3개 이상의 많은 창문",
                    "threshold": "창문 3개 이상",
                    "psychological_meaning": "HTP 기준 H24에 따르면 3개 이상의 많은 창문은 불안의 보상심리와 개방, 환경적 접촉에 대한 갈망을 나타냅니다. 이는 내적 불안을 외적 개방성으로 보상하려는 시도일 수 있습니다."
                })
    
    # 문 분석
    elif feature_name == "door":
        door_criteria = object_criteria.get("door", {})
        
        if feature_value == 0:
            if "missing" in door_criteria:
                criteria = door_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"문 크기 비율 {feature_value}으로 문이 완전히 없음",
                    "threshold": "문 0개 (완전 생략)",
                    "psychological_meaning": "HTP 기준 H22에 따르면 현관문이 생략된 집은 관계 회피, 고립, 위축을 나타냅니다. 이는 대인관계에서의 회피적 성향이나 사회적 고립을 의미합니다."
                })
        elif feature_value < 0.1:  # 매우 작은 문
            if "very_small" in door_criteria:
                criteria = door_criteria["very_small"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"문 크기 비율 {feature_value:.3f}이 임계값 0.1 미만으로 매우 작음",
                    "threshold": "문 크기 < 0.1",
                    "psychological_meaning": "HTP 기준 H19에 따르면 현관문이 집에 비해 과도하게 작은 경우 수줍음, 까다로움, 사회성 결핍, 현실도피를 나타냅니다. 이는 대인관계에서의 소극적 성향을 의미합니다."
                })
    
    # 굴뚝/연기 분석
    elif feature_name == "chimney":
        chimney_criteria = object_criteria.get("chimney", {})
        
        if feature_value == 1 or feature_value is True:
            if "with_smoke" in chimney_criteria:
                criteria = chimney_criteria["with_smoke"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"굴뚝 존재 여부 {feature_value}으로 굴뚝이 그려져 있음",
                    "threshold": "굴뚝 1개 (존재)",
                    "psychological_meaning": "HTP 기준 H27에 따르면 굴뚝의 연기 표현은 마음속 긴장, 가정 내 갈등, 정서 혼란을 나타냅니다. 이는 가정 내 불화나 내적 갈등의 표현일 수 있습니다."
                })
    
    # 나무 기둥 분석
    elif feature_name == "trunk" and isinstance(feature_value, (int, float)):
        trunk_criteria = object_criteria.get("trunk", {})
        
        if feature_value < 0.1:
            if "thin" in trunk_criteria:
                criteria = trunk_criteria["thin"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"나무 기둥 두께 비율 {feature_value:.3f}이 임계값 0.1 미만으로 매우 가늘음",
                    "threshold": "기둥 두께 < 0.1",
                    "psychological_meaning": "HTP 기준 T18에 따르면 나무기둥의 두께가 전체 나무 크기에 비해 얇은 경우 우울과 외로움을 나타냅니다. 이는 지지 기반의 약화나 불안정한 자아상을 의미합니다."
                })
    
    # 나무 가지 분석
    elif feature_name == "branches":
        branches_criteria = object_criteria.get("branches", {})
        
        if isinstance(feature_value, int):
            if feature_value >= 5:
                if "many" in branches_criteria:
                    criteria = branches_criteria["many"]
                    interpretation.update({
                        "interpretation": criteria["interpretation"],
                        "severity": criteria["severity"],
                        "score": criteria["score"],
                        "reasoning": f"가지 개수 {feature_value}개로 5개 이상의 많은 가지",
                        "threshold": "가지 5개 이상",
                        "psychological_meaning": "HTP 기준 T23에 따르면 수관에서 나뭇가지의 수가 지나치게 많은 표현은 하고 싶은 일이 많고, 대인관계가 활발하고 의욕이 과함을 나타냅니다. 이는 에너지와 활동성의 과도한 표현일 수 있습니다."
                    })
            elif feature_value <= 4:
                if "few" in branches_criteria:
                    criteria = branches_criteria["few"]
                    interpretation.update({
                        "interpretation": criteria["interpretation"],
                        "severity": criteria["severity"],
                        "score": criteria["score"],
                        "reasoning": f"가지 개수 {feature_value}개로 4개 이하의 적은 가지",
                        "threshold": "가지 4개 이하",
                        "psychological_meaning": "HTP 기준 T24에 따르면 수관에서 나뭇가지의 수가 4개 이하로 표현된 경우 세상과 상호작용에 억제적임, 위축과 우울감을 나타냅니다. 이는 사회적 활동의 제한이나 에너지 부족을 의미합니다."
                    })
    
    # 뿌리 분석
    elif feature_name == "roots":
        roots_criteria = object_criteria.get("roots", {})
        
        if feature_value == 1 or feature_value is True:
            if "underground_emphasized" in roots_criteria:
                criteria = roots_criteria["underground_emphasized"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"뿌리 존재 여부 {feature_value}으로 뿌리가 그려져 있음",
                    "threshold": "뿌리 1개 (존재)",
                    "psychological_meaning": "HTP 기준 T20에 따르면 땅속에 있는 뿌리를 강조하여 표현한 경우 현실적응의 장애, 예민함, 퇴행을 나타냅니다. 이는 안정감에 대한 과도한 욕구나 현실 도피 경향을 의미합니다."
                })
        elif feature_value == 0 or feature_value is False:
            if "exposed_no_ground" in roots_criteria:
                criteria = roots_criteria["exposed_no_ground"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"뿌리 존재 여부 {feature_value}으로 뿌리가 없음",
                    "threshold": "뿌리 0개 (없음)",
                    "psychological_meaning": "HTP 기준 T22에 따르면 지면선 없이 뿌리가 모두 노출된 표현은 유아기부터 지속된 불안, 우울의 표현을 나타냅니다. 이는 기반 부족이나 불안정한 정서 상태를 의미합니다."
                })
    
    # 잎 분석
    elif feature_name == "leaves" and isinstance(feature_value, (int, float)):
        leaves_criteria = object_criteria.get("leaves", {})
        
        if feature_value > 0.5:
            if "overly_detailed" in leaves_criteria:
                criteria = leaves_criteria["overly_detailed"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value:.3f}이 임계값 0.5 이상으로 과도하게 상세함",
                    "threshold": "잎 비율 > 0.5",
                    "psychological_meaning": "HTP 기준 T28에 따르면 수관의 잎이 구체적으로 과도하게 크게 표현된 경우 충동적, 정열, 희망적, 자신감(힘의 욕구 강화)을 나타냅니다. 이는 활력과 에너지의 과도한 표현일 수 있습니다."
                })
        elif feature_value < 0.2:
            if "fallen" in leaves_criteria:
                criteria = leaves_criteria["fallen"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value:.3f}이 임계값 0.2 미만으로 매우 적음",
                    "threshold": "잎 비율 < 0.2",
                    "psychological_meaning": "HTP 기준 T38에 따르면 떨어지거나 떨어진 잎의 표현은 우울, 외로움, 정서불안을 나타냅니다. 이는 활력 저하나 정서적 위축을 의미합니다."
                })
        elif feature_value == 0:
            if "bare_branches" in leaves_criteria:
                criteria = leaves_criteria["bare_branches"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value}으로 잎이 전혀 없음 (겨울나무)",
                    "threshold": "잎 비율 = 0",
                    "psychological_meaning": "HTP 기준 T16에 따르면 마른 가지만 있는 수관의 표현(겨울나무)은 자아 통제력 상실, 외상경험, 무력감, 수동적 성향을 나타냅니다. 이는 심리적 위축이나 에너지 부족을 의미합니다."
                })
    
    # 구멍 분석
    elif feature_name == "hole":
        holes_criteria = object_criteria.get("holes", {})
        
        if feature_value == 1 or feature_value is True:
            if "present" in holes_criteria:
                criteria = holes_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"]
                })
    
    # 사람 얼굴 분석
    elif feature_name == "face":
        face_criteria = object_criteria.get("face", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing_features" in face_criteria:
                criteria = face_criteria["missing_features"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"얼굴 특징 존재 여부 {feature_value}으로 얼굴 특징이 완전히 없음",
                    "threshold": "얼굴 특징 0개 (완전 생략)",
                    "psychological_meaning": "HTP 기준 P17에 따르면 얼굴의 눈, 코, 입이 생략된 경우 회피, 불안, 우울, 성적 갈등을 나타냅니다. 이는 정서표현 회피나 대인관계에서의 긴장을 의미합니다."
                })
    
    # 사람 손 분석
    elif feature_name == "hands":
        hands_criteria = object_criteria.get("hands", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing" in hands_criteria:
                criteria = hands_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"손 존재 여부 {feature_value}으로 손이 그려지지 않음",
                    "threshold": "손 0개 (생략)",
                    "psychological_meaning": "HTP 기준 P38에 따르면 팔이나 손의 생략은 죄의식, 우울, 무력감, 대인관계 기피, 과도한 업무를 나타냅니다. 이는 행동 통제의 어려움이나 사회적 유능감 저하를 의미합니다."
                })
        elif feature_value == 1 or feature_value is True:
            if "present" in hands_criteria:
                criteria = hands_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"손 존재 여부 {feature_value}으로 손이 그려져 있음",
                    "threshold": "손 1개 이상 (존재)",
                    "psychological_meaning": "손이 그려진 것은 행동 능력과 사회적 유능감을 나타냅니다. 이는 적극적인 행동 의지나 대인관계 능력을 의미할 수 있습니다."
                })
    
    # 사람 발 분석
    elif feature_name == "feet":
        legs_feet_criteria = object_criteria.get("legs_feet", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing" in legs_feet_criteria:
                criteria = legs_feet_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"발 존재 여부 {feature_value}으로 발이 그려지지 않음",
                    "threshold": "발 0개 (생략)",
                    "psychological_meaning": "HTP 기준 P43에 따르면 발을 표시하지 않은 경우나 절단된 다리 표현은 우울, 의기소침, 불안을 나타냅니다. 이는 현실 기반 부족이나 불안정한 정서 상태를 의미합니다."
                })
        elif feature_value == 1 or feature_value is True:
            if "present" in legs_feet_criteria:
                criteria = legs_feet_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"발 존재 여부 {feature_value}으로 발이 그려져 있음",
                    "threshold": "발 1개 이상 (존재)",
                    "psychological_meaning": "발이 그려진 것은 현실 기반과 안정감을 나타냅니다. 이는 현실적 지향이나 안정된 정서 상태를 의미할 수 있습니다."
                })
    
    return interpretation if interpretation["interpretation"] else None

def is_counseling_related(title, category, description):
    """상담센터 관련 키워드인지 판별"""
    
    # 상담센터 관련 키워드 (포함되어야 함)
    counseling_keywords = [
        '상담', '심리', '정신', '치료', '클리닉', '센터', '의원', '병원',
        '마음', '정신건강', '심리상담', '심리치료', '정신과', '정신건강복지',
        '상담센터', '심리상담센터', '심리치료센터', '정신건강복지센터',
        '심리클리닉', '마음상담센터', '정신과의원', '정신건강의학과',
        '우울', '불안', '스트레스', '트라우마', '가족상담', '부부상담',
        '청소년상담', '아동상담', '노인상담', '집단상담', '개인상담'
    ]
    
    # 제외할 키워드 (포함되면 안됨)
    exclude_keywords = [
        '카페', '커피', '음식점', '식당', '레스토랑', '패스트푸드',
        '가죽', '공방', '수제', '핸드메이드', '공예', '만들기',
        '미용', '헤어', '네일', '피부', '마사지', '스파',
        '헬스', '피트니스', '요가', '필라테스', '운동',
        '학원', '교육', '학습', '과외', '입시', '어학',
        '쇼핑', '마트', '편의점', '백화점', '상점',
        '호텔', '펜션', '모텔', '숙박', '여행',
        '은행', '보험', '금융', '증권', '대출',
        '자동차', '정비', '수리', '세차', '주유',
        '부동산', '중개', '임대', '매매', '분양'
    ]
    
    # 모든 텍스트를 소문자로 변환하여 검색
    text_to_check = f"{title} {category} {description}".lower()
    
    # 제외 키워드가 포함되어 있으면 False
    for exclude_keyword in exclude_keywords:
        if exclude_keyword in text_to_check:
            return False
    
    # 상담센터 관련 키워드가 하나라도 포함되어 있으면 True
    for counseling_keyword in counseling_keywords:
        if counseling_keyword in text_to_check:
            return True
    
    return False

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Base64 문자열을 이미지로 변환"""
    try:
        # data:image/png;base64, 부분 제거
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # RGB로 변환 (RGBA인 경우)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Base64 이미지 변환 오류: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    loaded_models = list(yolo_analyzer.models.keys())
    htp_criteria_count = len(htp_analyzer.htp_criteria)
    return jsonify({
        "status": "healthy",
        "message": "MindCanvas Backend is running",
        "loaded_models": loaded_models,
        "total_models": len(loaded_models),
        "htp_criteria_count": htp_criteria_count
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """이미지 분석 API"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "이미지 데이터가 필요합니다."
            }), 400
        
        # Base64 이미지 데이터 추출
        image_data = data['image']
        
        # Base64를 이미지로 변환
        image = base64_to_image(image_data)
        
        if image is None:
            return jsonify({
                "error": "이미지 변환에 실패했습니다."
            }), 400
        
        # YOLOv5로 객체 탐지
        house_detections = yolo_analyzer.predict(image, "House")
        
        # HTP 전문 분석기로 심리 분석
        analysis_result = htp_analyzer.analyze_house_drawing(house_detections)
        
        if analysis_result is None:
            return jsonify({
                "error": "이미지 분석에 실패했습니다."
            }), 500
        
        return jsonify({
            "success": True,
            "analysis": analysis_result,
            "message": "분석이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"분석 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록"""
    models_info = []
    for model_name, model_info in yolo_analyzer.models.items():
        models_info.append({
            "id": model_name.lower(),
            "name": f"{model_name} 분석",
            "description": f"{model_name} 그림을 분석하여 심리 상태를 파악합니다.",
            "status": "available",
            "classes": model_info["classes"]
        })
    
    return jsonify({
        "models": models_info,
        "htp_criteria_loaded": len(htp_analyzer.htp_criteria) > 0
    })

@app.route('/api/predict/<model_name>', methods=['POST'])
def predict_with_model(model_name):
    """특정 모델로 예측"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "이미지 데이터가 필요합니다."
            }), 400
        
        # Base64 이미지 데이터 추출
        image_data = data['image']
        
        # Base64를 이미지로 변환
        image = base64_to_image(image_data)
        
        if image is None:
            return jsonify({
                "error": "이미지 변환에 실패했습니다."
            }), 400
        
        # 모델로 예측
        detections = yolo_analyzer.predict(image, model_name)
        
        return jsonify({
            "success": True,
            "model": model_name,
            "detections": detections,
            "message": f"{model_name} 모델 분석이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"예측 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """HTP 전문 챗봇 API"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "메시지가 필요합니다."
            }), 400
        
        user_message = data['message']
        conversation_history = data.get('conversation_history', [])
        image_analysis_result = data.get('image_analysis_result', None)
        
        if not openai.api_key:
            return jsonify({
                "error": "OpenAI API 키가 설정되지 않았습니다."
            }), 500
        
        # HTP 전문 시스템 프롬프트 생성
        system_prompt = get_htp_system_prompt()
        
        # 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]
        
        # 기존 대화 기록 추가
        for msg in conversation_history:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                messages.append({"role": msg['role'], "content": msg['content']})
            elif isinstance(msg, tuple) and len(msg) == 2:
                messages.append({"role": "user", "content": msg[0]})
                messages.append({"role": "assistant", "content": msg[1]})
        
        # 이미지 분석 결과가 있으면 처리
        enhanced_query = user_message
        if image_analysis_result:
            analysis_result = analyze_image_features(image_analysis_result)
            
            if "error" not in analysis_result:
                analysis_summary = f"""
이미지 분석 결과:

총 점수: {analysis_result['total_score']}
위험도: {analysis_result['risk_level']}

객체별 분석:
"""
                
                for obj_id, obj_data in analysis_result['objects'].items():
                    analysis_summary += f"\n{obj_data['label']} (점수: {obj_data['score']}):\n"
                    for interpretation in obj_data['interpretations']:
                        analysis_summary += f"- {interpretation['feature']}: {interpretation['interpretation']} (심각도: {interpretation['severity']})\n"
                
                enhanced_query = f"{user_message}\n\n{analysis_summary}"
            else:
                enhanced_query = f"{user_message}\n\n이미지 분석 중 오류가 발생했습니다: {analysis_result['error']}"
        
        # 현재 질문 추가
        messages.append({"role": "user", "content": enhanced_query})
        
        # OpenAI API 호출 (최신 버전)
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        bot_response = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": bot_response,
            "message": "HTP 전문 챗봇 응답이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"챗봇 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

@app.route('/api/search', methods=['POST'])
def search_places():
    """네이버 검색 API 프록시"""
    try:
        data = request.get_json()
        query = data.get("query", "")
        display = data.get("display", 10)
        
        print(f"🔍 검색 요청 받음: {query}")
        
        if not query:
            return jsonify({"error": "검색어가 필요합니다"}), 400
        
        if not NAVER_SEARCH_CLIENT_ID or not NAVER_SEARCH_CLIENT_SECRET:
            return jsonify({"error": "네이버 검색 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://openapi.naver.com/v1/search/local.json",
                params={
                    "query": query,
                    "display": display,
                    "start": 1,
                    "sort": "random"
                },
                headers={
                    "X-Naver-Client-Id": NAVER_SEARCH_CLIENT_ID,
                    "X-Naver-Client-Secret": NAVER_SEARCH_CLIENT_SECRET,
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 검색 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            # 검색 결과 파싱 및 필터링
            if data.get("items"):
                results = []
                for item in data["items"]:
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    category = item.get("category", "")
                    description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                    
                    # 상담센터 관련 키워드 필터링 (일시적으로 비활성화)
                    print(f"🔍 검색 결과: {title} | 카테고리: {category} | 설명: {description}")
                    is_related = is_counseling_related(title, category, description)
                    print(f"🔍 필터링 결과: {is_related}")
                    
                    # 일시적으로 모든 결과를 포함 (디버깅용)
                    results.append({
                        "title": title,
                        "address": item.get("address", ""),
                        "roadAddress": item.get("roadAddress", ""),
                        "category": category,
                        "description": description,
                        "link": item.get("link", ""),
                        "telephone": item.get("telephone", ""),
                        "is_counseling_related": is_related  # 디버깅용 필드 추가
                    })
                
                return jsonify({
                    "success": True,
                    "data": results,
                    "total": data.get("total", 0),
                    "source": "naver_api"
                })
            else:
                return jsonify({
                    "success": True,
                    "data": [],
                    "total": 0,
                    "source": "naver_api"
                })
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """네이버 지오코딩 API 프록시"""
    try:
        data = request.get_json()
        address = data.get("address", "")
        
        print(f"🗺️ 지오코딩 요청 받음: {address}")
        
        if not address:
            return jsonify({"error": "주소가 필요합니다"}), 400
        
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return jsonify({"error": "네이버 지오코딩 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://maps.apigw.ntruss.com/map-geocode/v2/geocode",
                params={
                    "query": address,
                    "output": "json"
                },
                headers={
                    "x-ncp-apigw-api-key-id": NAVER_CLIENT_ID,
                    "x-ncp-apigw-api-key": NAVER_CLIENT_SECRET,
                    "Accept": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 지오코딩 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            if data.get("addresses") and len(data["addresses"]) > 0:
                address_info = data["addresses"][0]
                return jsonify({
                    "success": True,
                    "data": {
                        "lat": float(address_info.get("y", 0)),
                        "lng": float(address_info.get("x", 0)),
                        "address": address_info.get("roadAddress", ""),
                        "jibunAddress": address_info.get("jibunAddress", "")
                    },
                    "source": "naver_api"
                })
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """네이버 역지오코딩 API 프록시"""
    try:
        data = request.get_json()
        lat = data.get("lat")
        lng = data.get("lng")
        
        print(f"🗺️ 역지오코딩 요청 받음: {lat}, {lng}")
        
        if not lat or not lng:
            return jsonify({"error": "위도와 경도가 필요합니다"}), 400
        
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return jsonify({"error": "네이버 지오코딩 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc",
                params={
                    "coords": f"{lng},{lat}",
                    "output": "json"
                },
                headers={
                    "x-ncp-apigw-api-key-id": NAVER_CLIENT_ID,
                    "x-ncp-apigw-api-key": NAVER_CLIENT_SECRET,
                    "Accept": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 역지오코딩 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                region = result.get("region", {})
                land = result.get("land", {})
                
                address_parts = []
                if region.get("area1", {}).get("name"):
                    address_parts.append(region["area1"]["name"])
                if region.get("area2", {}).get("name"):
                    address_parts.append(region["area2"]["name"])
                if region.get("area3", {}).get("name"):
                    address_parts.append(region["area3"]["name"])
                
                full_address = " ".join(address_parts)
                
                return jsonify({
                    "success": True,
                    "data": {
                        "address": full_address,
                        "area1": region.get("area1", {}).get("name", ""),
                        "area2": region.get("area2", {}).get("name", ""),
                        "area3": region.get("area3", {}).get("name", ""),
                        "roadAddress": land.get("name", ""),
                        "jibunAddress": land.get("number1", "")
                    },
                    "source": "naver_api"
                })
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/drawings', methods=['POST'])
def save_drawing():
    """사용자의 그림과 분석 결과를 저장하는 API"""
    try:
        data = request.get_json()
        user_id = data.get('user_id') # 실제 구현에서는 JWT 등 인증 토큰에서 user_id를 가져와야 함
        image_data = data.get('image') # Base64 이미지 데이터
        analysis_result = data.get('analysis_result')

        # 디버깅: 수신된 데이터 로깅
        print(f"[save_drawing] 수신 user_id: {user_id}")
        print(f"[save_drawing] image_data 길이: {len(image_data) if image_data else 'None'}")
        print(f"[save_drawing] analysis_result 존재 여부: {analysis_result is not None}")

        if not user_id:
            return jsonify({"error": "사용자 ID가 필요합니다."}), 400
        if not image_data:
            return jsonify({"error": "이미지 데이터가 필요합니다."}), 400

        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "유효하지 않은 사용자 ID 형식입니다. 숫자를 기대합니다."}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": f"사용자 ID {user_id}에 해당하는 사용자를 찾을 수 없습니다."}), 404

        # Base64 이미지 데이터를 파일로 저장
        image = base64_to_image(image_data)
        if image is None:
            return jsonify({"error": "이미지 변환에 실패했습니다. 유효한 Base64 이미지 데이터를 제공해주세요."}), 400

        # 고유한 파일명 생성
        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        new_drawing = Drawing(
            user_id=user_id,
            image_path=image_path,
            analysis_result=analysis_result
        )
        db.session.add(new_drawing)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "그림과 분석 결과가 성공적으로 저장되었습니다.",
            "drawing_id": new_drawing.id
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f"그림 저장 API 오류: {e}")
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/drawings/<int:user_id>', methods=['GET'])
def get_user_drawings(user_id):
    """특정 사용자의 그림 및 분석 결과를 가져오는 API"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "유효하지 않은 사용자 ID입니다."}), 404

        drawings = Drawing.query.filter_by(user_id=user_id).order_by(Drawing.created_at.desc()).all()
        
        drawings_data = []
        for drawing in drawings:
            # 이미지 파일을 Base64로 다시 인코딩
            with open(drawing.image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            drawings_data.append({
                "id": drawing.id,
                "image": f"data:image/png;base64,{encoded_image}",
                "analysis_result": drawing.analysis_result,
                "created_at": drawing.created_at.isoformat(),
                "updated_at": drawing.updated_at.isoformat()
            })

        return jsonify({
            "success": True,
            "user_id": user_id,
            "drawings": drawings_data,
            "message": "사용자의 그림 목록을 성공적으로 가져왔습니다."
        }), 200

    except Exception as e:
        print(f"그림 조회 API 오류: {e}")
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/register', methods=['POST', 'OPTIONS'])
@cross_origin()
def register():
    """사용자 회원가입 API"""
    try:
        raw_data = request.get_data()
        # Debugging: print raw incoming data for analysis
        print(f"회원가입 API - Raw incoming data (bytes): {raw_data}")

        try:
            data_str = raw_data.decode('utf-8')
            data = json.loads(data_str)
            print(f"회원가입 API - Parsed JSON data: {data}")
        except UnicodeDecodeError as e:
            print(f"회원가입 API - UTF-8 디코딩 오류: {e}")
            return jsonify({"error": f"서버 오류: 데이터 디코딩 오류 - {str(e)}"}), 400
        except json.JSONDecodeError as e:
            print(f"회원가입 API - JSON 파싱 오류: {e}")
            return jsonify({"error": f"서버 오류: 유효한 JSON 형식이 아닙니다 - {str(e)}"}), 400
        except Exception as e:
            print(f"회원가입 API - 알 수 없는 데이터 처리 오류: {e}")
            return jsonify({"error": f"서버 오류: 알 수 없는 데이터 처리 오류 - {str(e)}"}), 500

        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if not username or not password:
            return jsonify({"error": "사용자 이름과 비밀번호가 필요합니다."}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password, email=email)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"success": True, "message": "회원가입이 성공적으로 완료되었습니다.", "user_id": new_user.id}), 201

    except Exception as e:
        db.session.rollback()
        print(f"회원가입 API 오류 (최상위 예외): {e}") # Added for debugging
        print("회원가입 API - 전체 트레이스백:\n" + traceback.format_exc())
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
@cross_origin()
def login():
    """사용자 로그인 API"""
    try:
        raw_data = request.get_data()
        # Debugging: print raw incoming data for analysis
        print(f"로그인 API - Raw incoming data (bytes): {raw_data}")

        try:
            data_str = raw_data.decode('utf-8')
            data = json.loads(data_str)
            print(f"로그인 API - Parsed JSON data: {data}")
        except UnicodeDecodeError as e:
            print(f"로그인 API - UTF-8 디코딩 오류: {e}")
            return jsonify({"error": f"서버 오류: 데이터 디코딩 오류 - {str(e)}"}), 400
        except json.JSONDecodeError as e:
            print(f"로그인 API - JSON 파싱 오류: {e}")
            return jsonify({"error": f"서버 오류: 유효한 JSON 형식이 아닙니다 - {str(e)}"}), 400
        except Exception as e:
            print(f"로그인 API - 알 수 없는 데이터 처리 오류: {e}")
            return jsonify({"error": f"서버 오류: 알 수 없는 데이터 처리 오류 - {str(e)}"}), 500

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "사용자 이름과 비밀번호가 필요합니다."}), 400

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            # JWT 토큰 생성
            token = generate_jwt_token(user.id, username)
            return jsonify({
                "success": True, 
                "message": "로그인이 성공적으로 완료되었습니다.", 
                "username": username,
                "user_id": user.id,
                "token": token
            }), 200
        else:
            return jsonify({"error": "잘못된 사용자 이름 또는 비밀번호입니다."}), 401

    except Exception as e:
        print(f"로그인 API 오류 (최상위 예외): {e}") # Added for debugging
        print("로그인 API - 전체 트레이스백:\n" + traceback.format_exc())
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/logout', methods=['POST', 'OPTIONS'])
@cross_origin()
@token_required
def logout():
    """사용자 로그아웃 API"""
    try:
        # JWT 토큰은 클라이언트에서 삭제하면 됩니다 (서버 측에서는 무효화할 필요 없음)
        return jsonify({
            "success": True, 
            "message": "로그아웃이 성공적으로 완료되었습니다."
        }), 200
    except Exception as e:
        print(f"로그아웃 API 오류: {e}")
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/verify-token', methods=['POST', 'OPTIONS'])
@cross_origin()
@token_required
def verify_token():
    """토큰 검증 API"""
    try:
        user_info = request.current_user
        return jsonify({
            "success": True,
            "message": "토큰이 유효합니다.",
            "user_id": user_info['user_id'],
            "username": user_info['username']
        }), 200
    except Exception as e:
        print(f"토큰 검증 API 오류: {e}")
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MindCanvas Backend 서버를 시작합니다...")
    print("=" * 60)
    print(f"로드된 YOLOv5 모델: {list(yolo_analyzer.models.keys())}")
    print(f"로드된 HTP 분석 기준: {len(htp_analyzer.htp_criteria)}개")
    print("서버 주소: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)