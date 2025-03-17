from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_mail import Mail, Message
from flask_socketio import SocketIO, emit, join_room, leave_room
from functools import wraps
from datetime import datetime, timedelta
import os
import json
import random
import uuid
import pandas as pd
import numpy as np
import threading
import sys
from werkzeug.utils import secure_filename

print("Starting web application...")
print("Checking required libraries...")

# 3. Kiểm tra thư viện cần thiết
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Using CPU for model.")
        
    # Kiểm tra Accelerate
    try:
        import accelerate
        print(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        print("WARNING: Accelerate library not found!")
        print("Install it with: pip install 'accelerate>=0.26.0'")
        
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install torch transformers")

# Tích hợp với mô hình Deepseek offline
class DeepseekAI:
    def __init__(self):
        # Cac cap do va chu de
        self.levels = ["Beginner (A1)", "Elementary (A2)", "Intermediate (B1)", 
                      "Upper Intermediate (B2)", "Advanced (C1)", "Proficient (C2)"]
        self.topics = ["Greetings", "Family", "Food", "Travel", "Work", "Hobbies", 
                      "Health", "Environment", "Technology", "Education", "Culture"]
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Duong dan den thu muc chua mo hinh Deepseek
            self.model_path = "./deepseek-model"
            
            # Thong bao log
            print("Loading Deepseek model from", self.model_path)
            
            # Tai tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Kiem tra GPU
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                print("Using GPU:", torch.cuda.get_device_name(0))
                torch_dtype = torch.float16
            else:
                print("Using CPU for model")
                torch_dtype = torch.float32
            
            # Kiem tra thu vien Accelerate
            has_accelerate = False
            try:
                import accelerate
                has_accelerate = True
                print("Accelerate library detected:", accelerate.__version__)
            except ImportError:
                print("Accelerate library not found, using basic loading")
            
            # Tai mo hinh dua vao co Accelerate hay khong
            if has_accelerate:
                # Neu co Accelerate, su dung cau hinh day du
                device_map = "auto" if has_gpu else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # Neu khong co Accelerate, tranh su dung cac tham so yeu cau Accelerate
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                
                # Di chuyen mo hinh den thiet bi phu hop
                if has_gpu:
                    self.model = self.model.to("cuda")
                else:
                    self.model = self.model.to("cpu")
            
            # Dat model vao che do evaluation
            self.model.eval()
            print("Model loaded successfully!")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading Deepseek model: {str(e)}")
            print("Switching to simulation mode...")
            self.model_loaded = False
    
    def _generate_response(self, prompt, max_length=2000, temperature=0.7):
        """Tao phan hoi tu mo hinh Deepseek"""
        try:
            if not self.model_loaded:
                raise Exception("Model not loaded")
                
            import torch
            
            # Tạo input cho model, có padding=True và truncation=True
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Chuyển inputs đến thiết bị phù hợp
            if hasattr(self.model, "device"):
                device = self.model.device
                for k, v in inputs.items():
                    if hasattr(v, "to") and callable(getattr(v, "to")):
                        inputs[k] = v.to(device)
            
            # Tạo văn bản với sampling
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],  # <-- chỉ cần thêm dòng này
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode kết quả
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Tách phần text sinh ra so với prompt
            prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            return full_text[len(prompt_text):]
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None


    
    def _extract_json_from_text(self, text):
        """Trich xuat JSON tu phan hoi cua mo hinh"""
        if not text:
            return None
            
        try:
            # Tim JSON trong van ban
            import re
            json_pattern = r'```json\n(.*?)\n```'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Thu tim trong dau ngoac nhon
            json_pattern = r'(\{.*\})'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
                
            # Khong tim thay JSON
            return None
            
        except Exception as e:
            print(f"Error extracting JSON: {str(e)}")
            return None
    
    def evaluate_test(self, answers, test_type="placement"):
        """Đánh giá bài kiểm tra và trả về mức độ thành thạo"""
        if self.model_loaded:
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp và nhiệm vụ của bạn là đánh giá trình độ tiếng Anh của học viên dựa trên bài kiểm tra. Hãy phân tích kết quả bài kiểm tra sau:

Loại bài kiểm tra: {test_type}
Các câu trả lời của học viên: {json.dumps(answers, ensure_ascii=False)}

Vui lòng đánh giá trình độ của học viên, chỉ trả về kết quả ở định dạng JSON theo cấu trúc sau:
```json
{{
  "level": "Cấp độ theo thang A1, A2, B1, B2, C1, C2",
  "score": "Điểm số từ 0-10",
  "strengths": ["điểm mạnh 1", "điểm mạnh 2", ...],
  "weaknesses": ["điểm yếu 1", "điểm yếu 2", ...],
  "recommendations": ["đề xuất 1", "đề xuất 2", ...]
}}
```
Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để phân tích
            response = self._generate_response(prompt)
            feedback = self._extract_json_from_text(response)
            
            if feedback:
                # Đảm bảo dữ liệu đúng định dạng
                if "score" in feedback and isinstance(feedback["score"], str):
                    try:
                        feedback["score"] = float(feedback["score"])
                    except:
                        feedback["score"] = 5.0
                return feedback
        
        # Nếu không thể sử dụng mô hình hoặc kết quả không hợp lệ, sử dụng phương pháp dự phòng
        # Mô phỏng đánh giá
        score = sum([ans.get("score", 0) for ans in answers]) / max(len(answers), 1)
        
        # Xác định cấp độ dựa trên điểm số
        level_index = min(int(score * len(self.levels) / 10), len(self.levels) - 1)
        level = self.levels[level_index]
        
        # Tạo phản hồi chi tiết
        feedback = {
            "level": level,
            "score": score,
            "strengths": ["Vocabulary" if score > 7 else "Basic grammar"],
            "weaknesses": ["Advanced grammar" if score < 8 else "Pronunciation"],
            "recommendations": [
                f"Nên tập trung vào {'ngữ pháp nâng cao' if score < 8 else 'từ vựng chuyên ngành'}",
                f"Thực hành {'nói' if score < 7 else 'viết'} thường xuyên"
            ]
        }
        return feedback
    
    def generate_test(self, level=None, topic=None, length=10):
        """Tạo bài kiểm tra dựa trên cấp độ và chủ đề"""
        if self.model_loaded:
            # Xác định loại bài kiểm tra
            test_type = "placement" if not level else "level"
            
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy tạo một bài kiểm tra tiếng Anh với các thông số sau:

Loại bài kiểm tra: {test_type}
{"Cấp độ: " + level if level else "Bài kiểm tra xếp loại từ cơ bản đến nâng cao"}
{"Chủ đề: " + topic if topic else ""}
Số lượng câu hỏi: {length}

Vui lòng tạo bài kiểm tra ở định dạng JSON theo cấu trúc sau:
```json
{{
  "questions": [
    {{
      "id": "q1",
      "type": "multiple_choice/writing/listening/speaking/grammar/vocabulary/reading/cloze",
      "question": "Nội dung câu hỏi",
      "options": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
      "difficulty": "Cấp độ của câu hỏi"
    }},
    ...
  ],
  "time_limit": "Thời gian làm bài tính bằng giây",
  "passing_score": "Điểm để đạt (từ 0-10)"
}}
```

Lưu ý:
- Với bài kiểm tra xếp loại, tạo câu hỏi có độ khó tăng dần từ A1 đến C2
- Đa dạng loại câu hỏi: trắc nghiệm, điền từ, đọc hiểu, nghe hiểu, ngữ pháp và từ vựng
- Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo bài kiểm tra
            response = self._generate_response(prompt)
            test_data = self._extract_json_from_text(response)
            
            if test_data and "questions" in test_data:
                # Đảm bảo các trường cần thiết tồn tại
                if "time_limit" not in test_data:
                    test_data["time_limit"] = 30 * length
                if "passing_score" not in test_data:
                    test_data["passing_score"] = 7.0
                return test_data
        
        # Nếu không thể sử dụng mô hình hoặc kết quả không hợp lệ, sử dụng phương pháp dự phòng
        if not level:
            # Bài kiểm tra xếp loại
            questions = []
            for i in range(length):
                level_idx = min(int(i / length * len(self.levels)), len(self.levels) - 1)
                difficulty = self.levels[level_idx]
                
                # Xen kẽ loại câu hỏi
                if i % 5 == 0:
                    # Câu hỏi nghe hiểu 
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "listening",
                        "audio_url": f"/static/audio/sample{i+1}.mp3",
                        "question": f"Hãy nghe đoạn hội thoại và chọn câu trả lời đúng:",
                        "options": [
                            "Option A - sample answer 1",
                            "Option B - sample answer 2",
                            "Option C - sample answer 3",
                            "Option D - sample answer 4"
                        ],
                        "difficulty": difficulty
                    })
                elif i % 5 == 1:
                    # Câu hỏi viết
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "writing",
                        "question": f"Viết một đoạn văn ngắn (3-5 câu) về chủ đề: {random.choice(self.topics)}",
                        "difficulty": difficulty
                    })
                elif i % 5 == 2:
                    # Câu hỏi nói
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "speaking",
                        "question": f"Hãy nói về {random.choice(self.topics)} trong 30 giây.",
                        "difficulty": difficulty
                    })
                else:
                    # Câu hỏi trắc nghiệm
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "multiple_choice",
                        "question": f"Câu hỏi mẫu số {i+1} cho cấp độ {difficulty}?",
                        "options": [
                            "Option A - sample answer 1",
                            "Option B - sample answer 2",
                            "Option C - sample answer 3",
                            "Option D - sample answer 4"
                        ],
                        "difficulty": difficulty
                    })
        else:
            # Bài kiểm tra theo cấp độ cụ thể
            questions = []
            for i in range(length):
                if i % 4 == 0:
                    # Câu hỏi ngữ pháp
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "grammar",
                        "question": f"Chọn đáp án đúng để hoàn thành câu sau: She ___ to the store yesterday.",
                        "options": ["go", "goes", "went", "going"],
                        "difficulty": level
                    })
                elif i % 4 == 1:
                    # Câu hỏi từ vựng
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "vocabulary",
                        "question": f"Chọn từ đồng nghĩa với 'happy':",
                        "options": ["sad", "joyful", "angry", "tired"],
                        "difficulty": level
                    })
                elif i % 4 == 2:
                    # Câu hỏi đọc hiểu
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "reading",
                        "text": "This is a sample reading passage. It contains several sentences that students need to read and understand to answer the question below.",
                        "question": "Đâu là ý chính của đoạn văn trên?",
                        "options": [
                            "Đây là một đoạn văn mẫu",
                            "Học sinh cần đọc nhiều sách",
                            "Câu trả lời C",
                            "Câu trả lời D"
                        ],
                        "difficulty": level
                    })
                else:
                    # Câu hỏi hoàn thành đoạn văn
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "cloze",
                        "question": "Điền từ thích hợp vào chỗ trống: I ___ a student.",
                        "options": ["am", "is", "are", "be"],
                        "difficulty": level
                    })
                    
        return {
            "questions": questions,
            "time_limit": 30 * length,  # 30 giây mỗi câu hỏi
            "passing_score": 7.0
        }
    
    def generate_lesson(self, level, topic=None, weakness=None):
        """Tạo bài học dựa trên cấp độ, chủ đề và điểm yếu"""
        if not topic:
            topic = random.choice(self.topics)
            
        if self.model_loaded:
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy tạo một bài học tiếng Anh với các thông số sau:

Cấp độ: {level}
Chủ đề: {topic}
{"Điểm yếu cần cải thiện: " + weakness if weakness else ""}

Vui lòng tạo bài học ở định dạng JSON theo cấu trúc sau:
```json
{{
  "title": "Tiêu đề bài học",
  "description": "Mô tả ngắn về bài học",
  "objectives": ["Mục tiêu 1", "Mục tiêu 2", ...],
  "sections": [
    {{
      "title": "Tiêu đề phần",
      "content": "Nội dung HTML của phần",
      "vocabulary_list": [
        {{"word": "từ mới", "meaning": "nghĩa", "example": "ví dụ sử dụng"}}
      ]
    }},
    {{
      "title": "Ngữ pháp",
      "content": "Giải thích ngữ pháp",
      "grammar_points": [
        {{
          "title": "Cấu trúc ngữ pháp",
          "explanation": "Giải thích",
          "examples": ["Ví dụ 1", "Ví dụ 2"]
        }}
      ]
    }},
    {{
      "title": "Bài tập luyện tập",
      "exercises": [
        {{
          "id": "ex1",
          "question": "Câu hỏi",
          "type": "loại bài tập",
          "options": ["Lựa chọn A", "Lựa chọn B", ...]
        }}
      ]
    }}
  ]
}}
```

Lưu ý:
- Tạo nội dung phù hợp với cấp độ {level}
- Tập trung vào chủ đề {topic}
- Bài học nên có ít nhất 3 phần: từ vựng, ngữ pháp và bài tập
- Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo bài học
            response = self._generate_response(prompt, max_length=4000)
            lesson_data = self._extract_json_from_text(response)
            
            if lesson_data:
                return lesson_data
                
        # Nếu không thể sử dụng mô hình hoặc kết quả không hợp lệ, sử dụng phương pháp dự phòng
        # Cấu trúc bài học
        lesson = {
            "title": f"{topic} for {level} Learners",
            "description": f"Bài học này giúp bạn nâng cao kiến thức về {topic} ở trình độ {level}.",
            "objectives": [
                f"Hiểu và sử dụng được từ vựng liên quan đến {topic}",
                f"Nắm vững các cấu trúc ngữ pháp thường dùng trong {topic}",
                "Phát triển kỹ năng nghe và nói trong tình huống thực tế"
            ],
            "sections": []
        }
        
        # Phần từ vựng
        vocabulary_section = {
            "title": "Từ vựng",
            "content": f"<p>Dưới đây là những từ vựng quan trọng về chủ đề {topic}:</p>",
            "vocabulary_list": []
        }
        
        # Tạo danh sách từ vựng giả định
        for i in range(10):
            vocabulary_section["vocabulary_list"].append({
                "word": f"word_{i}",
                "meaning": f"Nghĩa của từ {i}",
                "example": f"Đây là một ví dụ sử dụng từ {i} trong câu."
            })
        
        # Phần ngữ pháp
        grammar_section = {
            "title": "Ngữ pháp",
            "content": "<p>Cấu trúc ngữ pháp quan trọng:</p>",
            "grammar_points": [
                {
                    "title": "Cấu trúc 1",
                    "explanation": "Giải thích cấu trúc 1",
                    "examples": ["Ví dụ 1", "Ví dụ 2"]
                },
                {
                    "title": "Cấu trúc 2",
                    "explanation": "Giải thích cấu trúc 2",
                    "examples": ["Ví dụ 1", "Ví dụ 2"]
                }
            ]
        }
        
        # Phần luyện tập
        practice_section = {
            "title": "Bài tập luyện tập",
            "exercises": []
        }
        
        # Tạo các bài tập giả định
        for i in range(5):
            practice_section["exercises"].append({
                "id": f"ex{i+1}",
                "question": f"Câu hỏi luyện tập {i+1}",
                "type": random.choice(["multiple_choice", "fill_in_blank", "matching"]),
                "options": ["Option A", "Option B", "Option C", "Option D"] if i % 2 == 0 else None
            })
        
        # Thêm các phần vào bài học
        lesson["sections"].extend([vocabulary_section, grammar_section, practice_section])
        
        return lesson
    
    def generate_feedback(self, answers, correct_answers):
        """Tạo phản hồi chi tiết cho bài tập"""
        if self.model_loaded:
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy đánh giá bài tập của học viên với các thông tin sau:

Câu trả lời của học viên: {json.dumps(answers, ensure_ascii=False)}
Câu trả lời đúng: {json.dumps(correct_answers, ensure_ascii=False)}

Vui lòng đánh giá và đưa ra phản hồi ở định dạng JSON theo cấu trúc sau:
```json
{{
  "score": "Điểm số từ 0-10",
  "correct_count": "Số câu đúng",
  "total_questions": "Tổng số câu",
  "performance": "Đánh giá tổng quát (Xuất sắc/Tốt/Khá/Trung bình/Cần cải thiện)",
  "suggestions": ["Gợi ý cải thiện 1", "Gợi ý cải thiện 2", ...],
  "next_steps": ["Bước tiếp theo 1", "Bước tiếp theo 2", ...]
}}
```

Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo phản hồi
            response = self._generate_response(prompt)
            feedback_data = self._extract_json_from_text(response)
            
            if feedback_data:
                # Đảm bảo dữ liệu đúng định dạng
                if "score" in feedback_data and isinstance(feedback_data["score"], str):
                    try:
                        feedback_data["score"] = float(feedback_data["score"])
                    except:
                        feedback_data["score"] = 5.0
                return feedback_data
                
        # Nếu không thể sử dụng mô hình hoặc kết quả không hợp lệ, sử dụng phương pháp dự phòng
        correct_count = sum([1 for i, ans in enumerate(answers) if ans == correct_answers[i]])
        score = (correct_count / len(answers)) * 10 if answers else 0
        
        feedback = {
            "score": score,
            "correct_count": correct_count,
            "total_questions": len(answers),
            "performance": "Xuất sắc" if score >= 9 else "Tốt" if score >= 8 else "Khá" if score >= 7 else "Trung bình" if score >= 5 else "Cần cải thiện",
            "suggestions": [
                "Tiếp tục học tập chăm chỉ!" if score >= 8 else "Cần ôn tập lại các khái niệm cơ bản."
            ],
            "next_steps": [
                "Chuyển sang bài học tiếp theo" if score >= 7 else "Làm lại bài tập này",
                "Thử thách bản thân với bài tập khó hơn" if score >= 8 else "Xem lại lý thuyết"
            ]
        }
        
        return feedback

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///english_learning.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_email_password'

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo các extension
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Vui lòng đăng nhập để truy cập trang này.'
migrate = Migrate(app, db)
mail = Mail(app)
socketio = SocketIO(app)

# Khởi tạo AI
ai = DeepseekAI()

# Định nghĩa các mô hình dữ liệu
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    fullname = db.Column(db.String(120))
    role = db.Column(db.String(20), default='student')  # 'admin', 'teacher', 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    avatar = db.Column(db.String(255), default='default_avatar.png')
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    
    # Quan hệ
    profile = db.relationship('UserProfile', backref='user', uselist=False)
    test_results = db.relationship('TestResult', backref='user')
    course_enrollments = db.relationship('CourseEnrollment', backref='user')
    notifications = db.relationship('Notification', backref='user')
    forum_posts = db.relationship('ForumPost', backref='user')
    forum_comments = db.relationship('ForumComment', backref='user')
    badges = db.relationship('UserBadge', backref='user')
    
    def __repr__(self):
        return f'<User {self.username}>'

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    bio = db.Column(db.Text)
    language_level = db.Column(db.String(50))  # A1, A2, B1, B2, C1, C2
    learning_goals = db.Column(db.Text)
    birth_date = db.Column(db.Date)
    country = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    preferred_topics = db.Column(db.String(255))  # JSON string of topics
    study_streak = db.Column(db.Integer, default=0)
    total_points = db.Column(db.Integer, default=0)
    last_study_date = db.Column(db.Date)
    
    def __repr__(self):
        return f'<UserProfile for User #{self.user_id}>'

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    level = db.Column(db.String(50))  # A1, A2, B1, B2, C1, C2
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_published = db.Column(db.Boolean, default=False)
    image = db.Column(db.String(255), default='default_course.jpg')
    duration_weeks = db.Column(db.Integer)
    topic = db.Column(db.String(100))
    
    # Quan hệ
    creator = db.relationship('User', backref='created_courses')
    lessons = db.relationship('Lesson', backref='course')
    enrollments = db.relationship('CourseEnrollment', backref='course')
    
    def __repr__(self):
        return f'<Course {self.title}>'

class Lesson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text)  # HTML content or JSON
    order = db.Column(db.Integer)  # Thứ tự trong khóa học
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    estimated_time = db.Column(db.Integer)  # Minutes
    
    # Quan hệ
    exercises = db.relationship('Exercise', backref='lesson')
    materials = db.relationship('LearningMaterial', backref='lesson')
    
    def __repr__(self):
        return f'<Lesson {self.title}>'

class LearningMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50))  # video, pdf, audio, text
    file_path = db.Column(db.String(255))
    external_url = db.Column(db.String(255))
    description = db.Column(db.Text)
    is_required = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<LearningMaterial {self.title}>'

class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(50))  # quiz, writing, speaking, listening
    content = db.Column(db.Text)  # JSON format for questions
    time_limit = db.Column(db.Integer)  # Minutes
    points = db.Column(db.Integer, default=10)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quan hệ
    submissions = db.relationship('ExerciseSubmission', backref='exercise')
    
    def __repr__(self):
        return f'<Exercise {self.title}>'

class ExerciseSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercise.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answers = db.Column(db.Text)  # JSON format
    score = db.Column(db.Float)
    feedback = db.Column(db.Text)  # JSON feedback from AI
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    time_spent = db.Column(db.Integer)  # Seconds
    
    user = db.relationship('User', backref='exercise_submissions')
    
    def __repr__(self):
        return f'<ExerciseSubmission by User #{self.user_id} for Exercise #{self.exercise_id}>'

class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(50))  # placement, course, level
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    level = db.Column(db.String(50))  # A1, A2, B1, B2, C1, C2
    questions = db.Column(db.Text)  # JSON format
    time_limit = db.Column(db.Integer)  # Minutes
    passing_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quan hệ
    creator = db.relationship('User', backref='created_tests')
    results = db.relationship('TestResult', backref='test')
    
    def __repr__(self):
        return f'<Test {self.title}>'

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answers = db.Column(db.Text)  # JSON format
    score = db.Column(db.Float)
    level_result = db.Column(db.String(50))  # A1, A2, B1, B2, C1, C2
    feedback = db.Column(db.Text)  # JSON feedback from AI
    strengths = db.Column(db.Text)  # JSON array
    weaknesses = db.Column(db.Text)  # JSON array
    recommendations = db.Column(db.Text)  # JSON array
    time_spent = db.Column(db.Integer)  # Seconds
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TestResult by User #{self.user_id} for Test #{self.test_id}>'

class CourseEnrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    progress = db.Column(db.Float, default=0.0)  # 0-100%
    status = db.Column(db.String(50), default='active')  # active, completed, dropped
    
    def __repr__(self):
        return f'<CourseEnrollment by User #{self.user_id} for Course #{self.course_id}>'

class ForumCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    order = db.Column(db.Integer)
    
    # Quan hệ
    posts = db.relationship('ForumPost', backref='category')
    
    def __repr__(self):
        return f'<ForumCategory {self.name}>'

class ForumPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category_id = db.Column(db.Integer, db.ForeignKey('forum_category.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    views = db.Column(db.Integer, default=0)
    is_pinned = db.Column(db.Boolean, default=False)
    is_locked = db.Column(db.Boolean, default=False)
    
    # Quan hệ
    comments = db.relationship('ForumComment', backref='post')
    
    def __repr__(self):
        return f'<ForumPost {self.title}>'

class ForumComment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('forum_post.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    parent_id = db.Column(db.Integer, db.ForeignKey('forum_comment.id'))
    
    # Quan hệ cho comments lồng nhau
    replies = db.relationship('ForumComment', backref=db.backref('parent', remote_side=[id]))
    
    def __repr__(self):
        return f'<ForumComment by User #{self.user_id} on Post #{self.post_id}>'

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text)
    type = db.Column(db.String(50))  # system, course, test, forum, etc.
    is_read = db.Column(db.Boolean, default=False)
    url = db.Column(db.String(255))  # URL to redirect when clicked
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Notification for User #{self.user_id}: {self.title}>'

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    image = db.Column(db.String(255))
    requirement_type = db.Column(db.String(50))  # points, streak, tests, courses
    requirement_value = db.Column(db.Integer)
    
    # Quan hệ
    users = db.relationship('UserBadge', backref='badge')
    
    def __repr__(self):
        return f'<Badge {self.name}>'

class UserBadge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    badge_id = db.Column(db.Integer, db.ForeignKey('badge.id'), nullable=False)
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserBadge {self.badge_id} for User #{self.user_id}>'

class ChatRoom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_private = db.Column(db.Boolean, default=False)
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Quan hệ
    creator = db.relationship('User', backref='created_rooms')
    messages = db.relationship('ChatMessage', backref='room')
    
    def __repr__(self):
        return f'<ChatRoom {self.name}>'

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey('chat_room.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quan hệ
    user = db.relationship('User', backref='chat_messages')
    
    def __repr__(self):
        return f'<ChatMessage by User #{self.user_id} in Room #{self.room_id}>'

class Statistic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    study_time = db.Column(db.Integer, default=0)  # Minutes
    exercises_completed = db.Column(db.Integer, default=0)
    tests_completed = db.Column(db.Integer, default=0)
    points_earned = db.Column(db.Integer, default=0)
    
    # Quan hệ
    user = db.relationship('User', backref='statistics')
    
    def __repr__(self):
        return f'<Statistic for User #{self.user_id} on {self.date}>'

# Hàm userloader cho flask-login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Decorator kiểm tra quyền admin
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Bạn không có quyền truy cập trang này.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Decorator kiểm tra quyền giáo viên hoặc admin
def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['teacher', 'admin']:
            flash('Bạn không có quyền truy cập trang này.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Các route của ứng dụng
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Đăng nhập không thành công. Vui lòng kiểm tra tên đăng nhập và mật khẩu.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        fullname = request.form.get('fullname')
        
        if password != confirm_password:
            flash('Mật khẩu không khớp.', 'danger')
            return render_template('register.html')
        
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Tên đăng nhập hoặc email đã tồn tại.', 'danger')
            return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password, fullname=fullname)
        db.session.add(user)
        db.session.commit()
        
        # Tạo profile cho user
        profile = UserProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()
        
        flash('Đăng ký thành công! Bây giờ bạn có thể đăng nhập.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Lấy thông tin cá nhân
    user_profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    
    # Lấy khóa học đã đăng ký
    enrollments = CourseEnrollment.query.filter_by(user_id=current_user.id).all()
    courses = [enrollment.course for enrollment in enrollments]
    
    # Lấy kết quả bài test gần nhất
    recent_tests = TestResult.query.filter_by(user_id=current_user.id).order_by(TestResult.completed_at.desc()).limit(5).all()
    
    # Lấy thông báo chưa đọc
    notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False).order_by(Notification.created_at.desc()).limit(5).all()
    
    # Lấy bài học được gợi ý
    if user_profile and user_profile.language_level:
        # Nếu đã có level, tìm các khóa học phù hợp
        recommended_courses = Course.query.filter_by(level=user_profile.language_level, is_published=True).all()
    else:
        # Nếu chưa có level, gợi ý khóa học cho người mới bắt đầu
        recommended_courses = Course.query.filter_by(level="Beginner (A1)", is_published=True).all()
    
    return render_template('dashboard.html', 
                          user_profile=user_profile, 
                          courses=courses, 
                          recent_tests=recent_tests, 
                          notifications=notifications, 
                          recommended_courses=recommended_courses)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        current_user.fullname = request.form.get('fullname')
        
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file.filename != '':
                filename = secure_filename(f"{current_user.username}_{int(datetime.now().timestamp())}.jpg")
                avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                avatar_file.save(avatar_path)
                current_user.avatar = filename
        
        if profile:
            profile.bio = request.form.get('bio')
            profile.country = request.form.get('country')
            profile.phone = request.form.get('phone')
            if request.form.get('birth_date'):
                profile.birth_date = datetime.strptime(request.form.get('birth_date'), '%Y-%m-%d')
            profile.learning_goals = request.form.get('learning_goals')
            profile.preferred_topics = request.form.get('preferred_topics')
        
        db.session.commit()
        flash('Thông tin cá nhân đã được cập nhật thành công!', 'success')
        return redirect(url_for('profile'))
    
    # Lấy các huy hiệu của người dùng
    badges = UserBadge.query.filter_by(user_id=current_user.id).all()
    
    return render_template('profile.html', profile=profile, badges=badges)

@app.route('/test/placement')
@login_required
def placement_test():
    # Kiểm tra xem người dùng đã làm bài test xếp loại chưa
    existing_test = TestResult.query.filter_by(
        user_id=current_user.id
    ).join(Test).filter(Test.type == 'placement').first()
    
    if existing_test:
        flash('Bạn đã hoàn thành bài test xếp loại. Bạn có thể xem kết quả hoặc làm lại bài test.', 'info')
        return redirect(url_for('test_result', test_id=existing_test.test_id))
    
    # Tạo bài test xếp loại mới
    test_data = ai.generate_test()
    
    # Lưu test vào cơ sở dữ liệu
    test = Test(
        title="Bài test xếp loại",
        description="Bài test này sẽ giúp xác định trình độ tiếng Anh của bạn.",
        type="placement",
        questions=json.dumps(test_data["questions"]),
        time_limit=test_data["time_limit"] // 60,  # Chuyển đổi giây sang phút
        passing_score=test_data["passing_score"]
    )
    db.session.add(test)
    db.session.commit()
    
    return render_template('take_test.html', test=test, questions=test_data["questions"], time_limit=test_data["time_limit"])

@app.route('/test/<int:test_id>')
@login_required
def take_test(test_id):
    test = Test.query.get_or_404(test_id)
    questions = json.loads(test.questions)
    
    return render_template('take_test.html', test=test, questions=questions, time_limit=test.time_limit * 60)

@app.route('/test/<int:test_id>/submit', methods=['POST'])
@login_required
def submit_test(test_id):
    test = Test.query.get_or_404(test_id)
    
    # Lấy các câu trả lời từ form
    answers = []
    for key, value in request.form.items():
        if key.startswith('answer_'):
            question_id = key.split('_')[1]
            answers.append({
                "question_id": question_id,
                "answer": value,
                "score": float(request.form.get(f'score_{question_id}', 0))  # Điểm số từ frontend (mô phỏng)
            })
    
    # Đánh giá bài test bằng AI
    feedback = ai.evaluate_test(answers, test.type)
    
    # Lưu kết quả vào cơ sở dữ liệu
    test_result = TestResult(
        test_id=test.id,
        user_id=current_user.id,
        answers=json.dumps(answers),
        score=feedback["score"],
        level_result=feedback["level"],
        feedback=json.dumps(feedback),
        strengths=json.dumps(feedback["strengths"]),
        weaknesses=json.dumps(feedback["weaknesses"]),
        recommendations=json.dumps(feedback["recommendations"]),
        time_spent=int(request.form.get('time_spent', 0))
    )
    db.session.add(test_result)
    
    # Cập nhật trình độ trong profile
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if profile:
        profile.language_level = feedback["level"]
    
    db.session.commit()
    
    # Kiểm tra xem có nên cấp huy hiệu không
    check_and_award_badges(current_user.id)
    
    # Tạo thông báo
    notification = Notification(
        user_id=current_user.id,
        title="Bài test đã hoàn thành",
        content=f"Bạn đã hoàn thành bài test '{test.title}' với điểm số {feedback['score']:.1f}/10. Cấp độ của bạn là {feedback['level']}.",
        type="test",
        url=url_for('test_result', test_id=test.id)
    )
    db.session.add(notification)
    db.session.commit()
    
    return redirect(url_for('test_result', test_id=test.id))

@app.route('/test/<int:test_id>/result')
@login_required
def test_result(test_id):
    test = Test.query.get_or_404(test_id)
    result = TestResult.query.filter_by(test_id=test.id, user_id=current_user.id).order_by(TestResult.completed_at.desc()).first_or_404()
    
    feedback = json.loads(result.feedback)
    
    # Tạo gợi ý khóa học dựa trên kết quả
    recommended_courses = Course.query.filter_by(level=result.level_result, is_published=True).all()
    
    return render_template('test_result.html', test=test, result=result, feedback=feedback, recommended_courses=recommended_courses)

@app.route('/courses')
@login_required
def courses():
    # Lấy danh sách tất cả khóa học đã xuất bản
    published_courses = Course.query.filter_by(is_published=True).all()
    
    # Lấy danh sách khóa học đã đăng ký
    enrollments = CourseEnrollment.query.filter_by(user_id=current_user.id).all()
    enrolled_course_ids = [e.course_id for e in enrollments]
    
    # Phân loại khóa học theo trình độ
    beginner_courses = [c for c in published_courses if c.level.startswith('A')]
    intermediate_courses = [c for c in published_courses if c.level.startswith('B')]
    advanced_courses = [c for c in published_courses if c.level.startswith('C')]
    
    return render_template('courses.html', 
                          beginner_courses=beginner_courses,
                          intermediate_courses=intermediate_courses,
                          advanced_courses=advanced_courses,
                          enrolled_course_ids=enrolled_course_ids)

@app.route('/course/<int:course_id>')
@login_required
def course_detail(course_id):
    course = Course.query.get_or_404(course_id)
    lessons = Lesson.query.filter_by(course_id=course.id).order_by(Lesson.order).all()
    
    # Kiểm tra người dùng đã đăng ký khóa học chưa
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=course.id).first()
    
    return render_template('course_detail.html', course=course, lessons=lessons, enrollment=enrollment)

@app.route('/course/<int:course_id>/enroll')
@login_required
def enroll_course(course_id):
    course = Course.query.get_or_404(course_id)
    
    # Kiểm tra xem đã đăng ký chưa
    existing_enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=course.id).first()
    if existing_enrollment:
        flash('Bạn đã đăng ký khóa học này rồi.', 'info')
        return redirect(url_for('course_detail', course_id=course.id))
    
    # Tạo đăng ký mới
    enrollment = CourseEnrollment(
        user_id=current_user.id,
        course_id=course.id,
        last_accessed=datetime.utcnow()
    )
    db.session.add(enrollment)
    
    # Tạo thông báo
    notification = Notification(
        user_id=current_user.id,
        title="Đăng ký khóa học thành công",
        content=f"Bạn đã đăng ký thành công khóa học '{course.title}'.",
        type="course",
        url=url_for('course_detail', course_id=course.id)
    )
    db.session.add(notification)
    
    db.session.commit()
    
    flash(f'Bạn đã đăng ký thành công khóa học "{course.title}".', 'success')
    return redirect(url_for('course_detail', course_id=course.id))

@app.route('/lesson/<int:lesson_id>')
@login_required
def lesson_detail(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    course = Course.query.get_or_404(lesson.course_id)
    
    # Kiểm tra người dùng đã đăng ký khóa học chưa
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=course.id).first()
    if not enrollment:
        flash('Bạn chưa đăng ký khóa học này.', 'danger')
        return redirect(url_for('course_detail', course_id=course.id))
    
    # Cập nhật thời gian truy cập gần nhất
    enrollment.last_accessed = datetime.utcnow()
    db.session.commit()
    
    # Lấy tài liệu học tập
    materials = LearningMaterial.query.filter_by(lesson_id=lesson.id).all()
    
    # Lấy bài tập
    exercises = Exercise.query.filter_by(lesson_id=lesson.id).all()
    
    # Nếu nội dung bài học là JSON, parse nó
    try:
        content = json.loads(lesson.content)
    except:
        content = {"html": lesson.content}  # Nếu không phải JSON, coi như HTML
    
    return render_template('lesson_detail.html', lesson=lesson, course=course, 
                          materials=materials, exercises=exercises, content=content)

@app.route('/exercise/<int:exercise_id>')
@login_required
def take_exercise(exercise_id):
    exercise = Exercise.query.get_or_404(exercise_id)
    lesson = Lesson.query.get_or_404(exercise.lesson_id)
    
    # Kiểm tra đăng ký khóa học
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=lesson.course_id).first()
    if not enrollment:
        flash('Bạn chưa đăng ký khóa học này.', 'danger')
        return redirect(url_for('courses'))
    
    # Parse nội dung bài tập
    content = json.loads(exercise.content)
    
    return render_template('take_exercise.html', exercise=exercise, lesson=lesson, content=content)

@app.route('/exercise/<int:exercise_id>/submit', methods=['POST'])
@login_required
def submit_exercise(exercise_id):
    exercise = Exercise.query.get_or_404(exercise_id)
    
    # Lấy các câu trả lời từ form
    answers = []
    for key, value in request.form.items():
        if key.startswith('answer_'):
            question_id = key.split('_')[1]
            answers.append({
                "question_id": question_id,
                "answer": value
            })
    
    # Mô phỏng đáp án đúng
    correct_answers = ["Option A", "Option B", "Option C", "Option D"] * (len(answers) // 4 + 1)
    
    # Đánh giá bài tập bằng AI
    feedback = ai.generate_feedback(
        [a["answer"] for a in answers], 
        correct_answers[:len(answers)]
    )
    
    # Lưu kết quả vào cơ sở dữ liệu
    submission = ExerciseSubmission(
        exercise_id=exercise.id,
        user_id=current_user.id,
        answers=json.dumps(answers),
        score=feedback["score"],
        feedback=json.dumps(feedback),
        time_spent=int(request.form.get('time_spent', 0))
    )
    db.session.add(submission)
    
    # Cập nhật tiến độ khóa học
    lesson = Lesson.query.get(exercise.lesson_id)
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=lesson.course_id).first()
    if enrollment:
        # Tính toán tiến độ
        total_exercises = Exercise.query.join(Lesson).filter(Lesson.course_id == lesson.course_id).count()
        completed_exercises = ExerciseSubmission.query.join(Exercise).join(Lesson).filter(
            ExerciseSubmission.user_id == current_user.id,
            Lesson.course_id == lesson.course_id
        ).distinct(ExerciseSubmission.exercise_id).count()
        
        enrollment.progress = min(100, (completed_exercises / total_exercises) * 100 if total_exercises > 0 else 0)
    
    # Cập nhật thống kê
    today = datetime.utcnow().date()
    stat = Statistic.query.filter_by(user_id=current_user.id, date=today).first()
    if not stat:
        stat = Statistic(user_id=current_user.id, date=today)
        db.session.add(stat)
    
    stat.exercises_completed += 1
    stat.points_earned += int(feedback["score"])
    
    # Cập nhật tổng điểm trong hồ sơ
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if profile:
        profile.total_points += int(feedback["score"])
    
    db.session.commit()
    
    # Kiểm tra và trao huy hiệu
    check_and_award_badges(current_user.id)
    
    return redirect(url_for('exercise_result', submission_id=submission.id))

@app.route('/exercise/result/<int:submission_id>')
@login_required
def exercise_result(submission_id):
    submission = ExerciseSubmission.query.filter_by(id=submission_id, user_id=current_user.id).first_or_404()
    exercise = Exercise.query.get_or_404(submission.exercise_id)
    
    answers = json.loads(submission.answers)
    feedback = json.loads(submission.feedback)
    
    return render_template('exercise_result.html', 
                          submission=submission, 
                          exercise=exercise, 
                          answers=answers, 
                          feedback=feedback)

@app.route('/forums')
@login_required
def forums():
    categories = ForumCategory.query.order_by(ForumCategory.order).all()
    
    # Lấy các bài viết đã ghim
    pinned_posts = ForumPost.query.filter_by(is_pinned=True).all()
    
    # Lấy các bài viết mới nhất
    recent_posts = ForumPost.query.order_by(ForumPost.created_at.desc()).limit(5).all()
    
    return render_template('forums.html', categories=categories, pinned_posts=pinned_posts, recent_posts=recent_posts)

@app.route('/forum/category/<int:category_id>')
@login_required
def forum_category(category_id):
    category = ForumCategory.query.get_or_404(category_id)
    posts = ForumPost.query.filter_by(category_id=category.id).order_by(ForumPost.is_pinned.desc(), ForumPost.created_at.desc()).all()
    
    return render_template('forum_category.html', category=category, posts=posts)

@app.route('/forum/post/<int:post_id>')
@login_required
def forum_post(post_id):
    post = ForumPost.query.get_or_404(post_id)
    
    # Tăng lượt xem
    post.views += 1
    db.session.commit()
    
    # Lấy tất cả comments
    comments = ForumComment.query.filter_by(post_id=post.id, parent_id=None).order_by(ForumComment.created_at).all()
    
    return render_template('forum_post.html', post=post, comments=comments)

@app.route('/forum/post/create', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category_id = request.form.get('category_id')
        
        post = ForumPost(
            title=title,
            content=content,
            category_id=category_id,
            user_id=current_user.id
        )
        db.session.add(post)
        db.session.commit()
        
        flash('Bài viết đã được tạo thành công!', 'success')
        return redirect(url_for('forum_post', post_id=post.id))
    
    categories = ForumCategory.query.all()
    return render_template('create_post.html', categories=categories)

@app.route('/forum/post/<int:post_id>/comment', methods=['POST'])
@login_required
def add_comment(post_id):
    post = ForumPost.query.get_or_404(post_id)
    
    content = request.form.get('content')
    parent_id = request.form.get('parent_id')
    
    comment = ForumComment(
        content=content,
        post_id=post.id,
        user_id=current_user.id,
        parent_id=parent_id if parent_id else None
    )
    db.session.add(comment)
    
    # Thông báo cho chủ bài viết nếu không phải là người comment
    if post.user_id != current_user.id:
        notification = Notification(
            user_id=post.user_id,
            title="Có bình luận mới trên bài viết của bạn",
            content=f"{current_user.username} đã bình luận trên bài viết '{post.title}'.",
            type="forum",
            url=url_for('forum_post', post_id=post.id)
        )
        db.session.add(notification)
    
    db.session.commit()
    
    return redirect(url_for('forum_post', post_id=post.id))

@app.route('/chat')
@login_required
def chat_rooms():
    public_rooms = ChatRoom.query.filter_by(is_private=False).all()
    
    return render_template('chat_rooms.html', rooms=public_rooms)

@app.route('/chat/<int:room_id>')
@login_required
def chat_room(room_id):
    room = ChatRoom.query.get_or_404(room_id)
    
    # Lấy 50 tin nhắn gần nhất
    messages = ChatMessage.query.filter_by(room_id=room.id).order_by(ChatMessage.created_at).limit(50).all()
    
    return render_template('chat_room.html', room=room, messages=messages)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    
    # Thông báo cho mọi người trong phòng
    emit('status', {'msg': f'{current_user.username} đã tham gia phòng chat'}, room=room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)
    
    # Thông báo cho mọi người trong phòng
    emit('status', {'msg': f'{current_user.username} đã rời phòng chat'}, room=room)

@socketio.on('message')
def on_message(data):
    room_id = data['room']
    content = data['message']
    
    # Lưu tin nhắn vào cơ sở dữ liệu
    message = ChatMessage(
        room_id=room_id,
        user_id=current_user.id,
        content=content
    )
    db.session.add(message)
    db.session.commit()
    
    # Gửi tin nhắn cho tất cả người dùng trong phòng
    emit('message', {
        'user': current_user.username,
        'avatar': current_user.avatar,
        'content': content,
        'time': datetime.utcnow().strftime('%H:%M')
    }, room=room_id)

@app.route('/notifications')
@login_required
def notifications():
    # Lấy tất cả thông báo, sắp xếp theo thời gian
    all_notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).all()
    
    return render_template('notifications.html', notifications=all_notifications)

@app.route('/notification/<int:notification_id>/read')
@login_required
def read_notification(notification_id):
    notification = Notification.query.filter_by(id=notification_id, user_id=current_user.id).first_or_404()
    
    notification.is_read = True
    db.session.commit()
    
    # Chuyển hướng đến URL trong thông báo nếu có
    if notification.url:
        return redirect(notification.url)
    
    return redirect(url_for('notifications'))

@app.route('/leaderboard')
@login_required
def leaderboard():
    # Lấy top 10 người dùng có điểm cao nhất
    top_users = db.session.query(User, UserProfile).join(UserProfile).order_by(UserProfile.total_points.desc()).limit(10).all()
    
    # Xác định thứ hạng của người dùng hiện tại
    current_rank = db.session.query(
        db.func.count(UserProfile.id)
    ).filter(
        UserProfile.total_points > UserProfile.query.filter_by(user_id=current_user.id).first().total_points
    ).scalar() + 1
    
    return render_template('leaderboard.html', top_users=top_users, current_rank=current_rank)

@app.route('/badges')
@login_required
def badges():
    # Lấy tất cả huy hiệu
    all_badges = Badge.query.all()
    
    # Lấy huy hiệu của người dùng
    user_badges = UserBadge.query.filter_by(user_id=current_user.id).all()
    earned_badge_ids = [ub.badge_id for ub in user_badges]
    
    return render_template('badges.html', badges=all_badges, earned_badge_ids=earned_badge_ids)

@app.route('/statistics')
@login_required
def user_statistics():
    # Lấy thống kê học tập của 7 ngày gần nhất
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    stats = Statistic.query.filter(
        Statistic.user_id == current_user.id,
        Statistic.date >= week_ago
    ).order_by(Statistic.date).all()
    
    # Tổng hợp thống kê
    total_study_time = sum([s.study_time for s in stats])
    total_exercises = sum([s.exercises_completed for s in stats])
    total_tests = sum([s.tests_completed for s in stats])
    total_points = sum([s.points_earned for s in stats])
    
    # Lấy kết quả bài test gần đây
    recent_tests = TestResult.query.filter_by(user_id=current_user.id).order_by(TestResult.completed_at.desc()).limit(5).all()
    
    return render_template('statistics.html', 
                          stats=stats, 
                          total_study_time=total_study_time,
                          total_exercises=total_exercises,
                          total_tests=total_tests,
                          total_points=total_points,
                          recent_tests=recent_tests)

# Admin routes
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    # Thống kê cơ bản
    total_users = User.query.count()
    total_courses = Course.query.count()
    total_lessons = Lesson.query.count()
    total_tests = Test.query.count()
    
    # 10 người dùng mới nhất
    latest_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    # Các khóa học mới tạo
    latest_courses = Course.query.order_by(Course.created_at.desc()).limit(5).all()
    
    return render_template('admin/dashboard.html', 
                          total_users=total_users,
                          total_courses=total_courses,
                          total_lessons=total_lessons,
                          total_tests=total_tests,
                          latest_users=latest_users,
                          latest_courses=latest_courses)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        user.username = request.form.get('username')
        user.email = request.form.get('email')
        user.fullname = request.form.get('fullname')
        user.role = request.form.get('role')
        user.is_active = True if request.form.get('is_active') else False
        
        if request.form.get('password'):
            user.password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        
        db.session.commit()
        flash('Thông tin người dùng đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/edit_user.html', user=user)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('Bạn không thể xóa tài khoản của chính mình!', 'danger')
        return redirect(url_for('admin_users'))
    
    db.session.delete(user)
    db.session.commit()
    
    flash('Người dùng đã được xóa thành công!', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/courses')
@login_required
@admin_required
def admin_courses():
    courses = Course.query.all()
    return render_template('admin/courses.html', courses=courses)

@app.route('/admin/course/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_course():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        level = request.form.get('level')
        duration_weeks = request.form.get('duration_weeks')
        topic = request.form.get('topic')
        
        course = Course(
            title=title,
            description=description,
            level=level,
            creator_id=current_user.id,
            duration_weeks=duration_weeks,
            topic=topic
        )
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"course_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                course.image = filename
        
        db.session.add(course)
        db.session.commit()
        
        flash('Khóa học đã được tạo thành công!', 'success')
        return redirect(url_for('admin_courses'))
    
    return render_template('admin/create_course.html')

@app.route('/admin/course/<int:course_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_course(course_id):
    course = Course.query.get_or_404(course_id)
    
    if request.method == 'POST':
        course.title = request.form.get('title')
        course.description = request.form.get('description')
        course.level = request.form.get('level')
        course.duration_weeks = request.form.get('duration_weeks')
        course.topic = request.form.get('topic')
        course.is_published = True if request.form.get('is_published') else False
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"course_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                course.image = filename
        
        db.session.commit()
        flash('Khóa học đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_courses'))
    
    return render_template('admin/edit_course.html', course=course)

@app.route('/admin/course/<int:course_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_course(course_id):
    course = Course.query.get_or_404(course_id)
    db.session.delete(course)
    db.session.commit()
    
    flash('Khóa học đã được xóa thành công!', 'success')
    return redirect(url_for('admin_courses'))

@app.route('/admin/course/<int:course_id>/lessons')
@login_required
@admin_required
def admin_course_lessons(course_id):
    course = Course.query.get_or_404(course_id)
    lessons = Lesson.query.filter_by(course_id=course.id).order_by(Lesson.order).all()
    
    return render_template('admin/course_lessons.html', course=course, lessons=lessons)

@app.route('/admin/course/<int:course_id>/lesson/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_lesson(course_id):
    course = Course.query.get_or_404(course_id)
    
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        order = request.form.get('order')
        estimated_time = request.form.get('estimated_time')
        
        # Tạo nội dung bài học từ AI nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            ai_content = ai.generate_lesson(course.level, course.topic)
            content = json.dumps(ai_content)
        
        lesson = Lesson(
            course_id=course.id,
            title=title,
            content=content,
            order=order,
            estimated_time=estimated_time
        )
        db.session.add(lesson)
        db.session.commit()
        
        flash('Bài học đã được tạo thành công!', 'success')
        return redirect(url_for('admin_course_lessons', course_id=course.id))
    
    # Xác định thứ tự mặc định cho bài học mới
    next_order = db.session.query(db.func.max(Lesson.order)).filter(Lesson.course_id == course.id).scalar()
    next_order = 1 if next_order is None else next_order + 1
    
    return render_template('admin/create_lesson.html', course=course, next_order=next_order)

@app.route('/admin/lesson/<int:lesson_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    
    if request.method == 'POST':
        lesson.title = request.form.get('title')
        lesson.content = request.form.get('content')
        lesson.order = request.form.get('order')
        lesson.estimated_time = request.form.get('estimated_time')
        
        # Tạo lại nội dung bài học từ AI nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            course = Course.query.get(lesson.course_id)
            ai_content = ai.generate_lesson(course.level, course.topic)
            lesson.content = json.dumps(ai_content)
        
        lesson.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash('Bài học đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_course_lessons', course_id=lesson.course_id))
    
    return render_template('admin/edit_lesson.html', lesson=lesson)

@app.route('/admin/lesson/<int:lesson_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    course_id = lesson.course_id
    
    db.session.delete(lesson)
    db.session.commit()
    
    flash('Bài học đã được xóa thành công!', 'success')
    return redirect(url_for('admin_course_lessons', course_id=course_id))

@app.route('/admin/lesson/<int:lesson_id>/exercises')
@login_required
@admin_required
def admin_lesson_exercises(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    exercises = Exercise.query.filter_by(lesson_id=lesson.id).all()
    
    return render_template('admin/lesson_exercises.html', lesson=lesson, exercises=exercises)

@app.route('/admin/lesson/<int:lesson_id>/exercise/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_exercise(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        exercise_type = request.form.get('type')
        time_limit = request.form.get('time_limit')
        points = request.form.get('points')
        
        # Tạo nội dung bài tập từ AI
        if request.form.get('generate_content') == 'yes':
            course = Course.query.get(lesson.course_id)
            ai_content = ai.generate_test(course.level, exercise_type, int(request.form.get('num_questions', 10)))
            content = json.dumps(ai_content["questions"])
        else:
            content = request.form.get('content')
        
        exercise = Exercise(
            lesson_id=lesson.id,
            title=title,
            description=description,
            type=exercise_type,
            content=content,
            time_limit=time_limit,
            points=points
        )
        db.session.add(exercise)
        db.session.commit()
        
        flash('Bài tập đã được tạo thành công!', 'success')
        return redirect(url_for('admin_lesson_exercises', lesson_id=lesson.id))
    
    return render_template('admin/create_exercise.html', lesson=lesson)

@app.route('/admin/exercise/<int:exercise_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_exercise(exercise_id):
    exercise = Exercise.query.get_or_404(exercise_id)
    
    if request.method == 'POST':
        exercise.title = request.form.get('title')
        exercise.description = request.form.get('description')
        exercise.type = request.form.get('type')
        exercise.time_limit = request.form.get('time_limit')
        exercise.points = request.form.get('points')
        
        # Tạo lại nội dung bài tập từ AI
        if request.form.get('generate_content') == 'yes':
            lesson = Lesson.query.get(exercise.lesson_id)
            course = Course.query.get(lesson.course_id)
            ai_content = ai.generate_test(course.level, exercise.type, int(request.form.get('num_questions', 10)))
            exercise.content = json.dumps(ai_content["questions"])
        else:
            exercise.content = request.form.get('content')
        
        db.session.commit()
        
        flash('Bài tập đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_lesson_exercises', lesson_id=exercise.lesson_id))
    
    return render_template('admin/edit_exercise.html', exercise=exercise)

@app.route('/admin/exercise/<int:exercise_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_exercise(exercise_id):
    exercise = Exercise.query.get_or_404(exercise_id)
    lesson_id = exercise.lesson_id
    
    db.session.delete(exercise)
    db.session.commit()
    
    flash('Bài tập đã được xóa thành công!', 'success')
    return redirect(url_for('admin_lesson_exercises', lesson_id=lesson_id))

@app.route('/admin/tests')
@login_required
@admin_required
def admin_tests():
    tests = Test.query.all()
    return render_template('admin/tests.html', tests=tests)

@app.route('/admin/test/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_test():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        test_type = request.form.get('type')
        level = request.form.get('level') if request.form.get('type') != 'placement' else None
        time_limit = request.form.get('time_limit')
        passing_score = request.form.get('passing_score')
        
        # Tạo nội dung bài test từ AI
        if test_type == 'placement':
            ai_content = ai.generate_test()
        else:
            ai_content = ai.generate_test(level, None, int(request.form.get('num_questions', 10)))
        
        test = Test(
            title=title,
            description=description,
            type=test_type,
            creator_id=current_user.id,
            level=level,
            questions=json.dumps(ai_content["questions"]),
            time_limit=time_limit,
            passing_score=passing_score
        )
        db.session.add(test)
        db.session.commit()
        
        flash('Bài test đã được tạo thành công!', 'success')
        return redirect(url_for('admin_tests'))
    
    return render_template('admin/create_test.html')

@app.route('/admin/test/<int:test_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_test(test_id):
    test = Test.query.get_or_404(test_id)
    
    if request.method == 'POST':
        test.title = request.form.get('title')
        test.description = request.form.get('description')
        test.type = request.form.get('type')
        if test.type != 'placement':
            test.level = request.form.get('level')
        test.time_limit = request.form.get('time_limit')
        test.passing_score = request.form.get('passing_score')
        
        # Tạo lại nội dung bài test từ AI nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            if test.type == 'placement':
                ai_content = ai.generate_test()
            else:
                ai_content = ai.generate_test(test.level, None, int(request.form.get('num_questions', 10)))
            test.questions = json.dumps(ai_content["questions"])
        
        db.session.commit()
        
        flash('Bài test đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_tests'))
    
    return render_template('admin/edit_test.html', test=test)

@app.route('/admin/test/<int:test_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_test(test_id):
    test = Test.query.get_or_404(test_id)
    db.session.delete(test)
    db.session.commit()
    
    flash('Bài test đã được xóa thành công!', 'success')
    return redirect(url_for('admin_tests'))

@app.route('/admin/forum-categories')
@login_required
@admin_required
def admin_forum_categories():
    categories = ForumCategory.query.order_by(ForumCategory.order).all()
    return render_template('admin/forum_categories.html', categories=categories)

@app.route('/admin/forum-category/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_forum_category():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        order = request.form.get('order')
        
        category = ForumCategory(
            name=name,
            description=description,
            order=order
        )
        db.session.add(category)
        db.session.commit()
        
        flash('Danh mục diễn đàn đã được tạo thành công!', 'success')
        return redirect(url_for('admin_forum_categories'))
    
    # Xác định thứ tự mặc định cho danh mục mới
    next_order = db.session.query(db.func.max(ForumCategory.order)).scalar()
    next_order = 1 if next_order is None else next_order + 1
    
    return render_template('admin/create_forum_category.html', next_order=next_order)

@app.route('/admin/forum-category/<int:category_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_forum_category(category_id):
    category = ForumCategory.query.get_or_404(category_id)
    
    if request.method == 'POST':
        category.name = request.form.get('name')
        category.description = request.form.get('description')
        category.order = request.form.get('order')
        
        db.session.commit()
        
        flash('Danh mục diễn đàn đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_forum_categories'))
    
    return render_template('admin/edit_forum_category.html', category=category)

@app.route('/admin/forum-category/<int:category_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_forum_category(category_id):
    category = ForumCategory.query.get_or_404(category_id)
    db.session.delete(category)
    db.session.commit()
    
    flash('Danh mục diễn đàn đã được xóa thành công!', 'success')
    return redirect(url_for('admin_forum_categories'))

@app.route('/admin/badges')
@login_required
@admin_required
def admin_badges():
    badges = Badge.query.all()
    return render_template('admin/badges.html', badges=badges)

@app.route('/admin/badge/create', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_create_badge():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        requirement_type = request.form.get('requirement_type')
        requirement_value = request.form.get('requirement_value')
        
        badge = Badge(
            name=name,
            description=description,
            requirement_type=requirement_type,
            requirement_value=requirement_value
        )
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"badge_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                badge.image = filename
        
        db.session.add(badge)
        db.session.commit()
        
        flash('Huy hiệu đã được tạo thành công!', 'success')
        return redirect(url_for('admin_badges'))
    
    return render_template('admin/create_badge.html')

@app.route('/admin/badge/<int:badge_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_badge(badge_id):
    badge = Badge.query.get_or_404(badge_id)
    
    if request.method == 'POST':
        badge.name = request.form.get('name')
        badge.description = request.form.get('description')
        badge.requirement_type = request.form.get('requirement_type')
        badge.requirement_value = request.form.get('requirement_value')
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"badge_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                badge.image = filename
        
        db.session.commit()
        
        flash('Huy hiệu đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_badges'))
    
    return render_template('admin/edit_badge.html', badge=badge)

@app.route('/admin/badge/<int:badge_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_badge(badge_id):
    badge = Badge.query.get_or_404(badge_id)
    db.session.delete(badge)
    db.session.commit()
    
    flash('Huy hiệu đã được xóa thành công!', 'success')
    return redirect(url_for('admin_badges'))

@app.route('/admin/reports')
@login_required
@admin_required
def admin_reports():
    # Số người dùng mới đăng ký trong 30 ngày qua
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    new_users_count = User.query.filter(User.created_at >= thirty_days_ago).count()
    
    # Tổng số bài test đã hoàn thành
    total_test_results = TestResult.query.count()
    
    # Tổng số khóa học đã đăng ký
    total_enrollments = CourseEnrollment.query.count()
    
    # Tổng thời gian học tập (phút)
    total_study_time = db.session.query(db.func.sum(Statistic.study_time)).scalar() or 0
    
    # Các cấp độ người dùng
    user_levels = db.session.query(
        UserProfile.language_level,
        db.func.count(UserProfile.id)
    ).group_by(UserProfile.language_level).all()
    
    return render_template('admin/reports.html',
                          new_users_count=new_users_count,
                          total_test_results=total_test_results,
                          total_enrollments=total_enrollments,
                          total_study_time=total_study_time,
                          user_levels=user_levels)

# Teacher routes
@app.route('/teacher/dashboard')
@login_required
@teacher_required
def teacher_dashboard():
    # Các khóa học do giáo viên tạo
    courses = Course.query.filter_by(creator_id=current_user.id).all()
    
    # Các bài test do giáo viên tạo
    tests = Test.query.filter_by(creator_id=current_user.id).all()
    
    # Tổng số học sinh tham gia các khóa học của giáo viên
    student_count = db.session.query(
        db.func.count(db.distinct(CourseEnrollment.user_id))
    ).join(Course).filter(Course.creator_id == current_user.id).scalar() or 0
    
    return render_template('teacher/dashboard.html', 
                          courses=courses, 
                          tests=tests, 
                          student_count=student_count)

@app.route('/teacher/courses')
@login_required
@teacher_required
def teacher_courses():
    # Các khóa học do giáo viên tạo
    courses = Course.query.filter_by(creator_id=current_user.id).all()
    return render_template('teacher/courses.html', courses=courses)

@app.route('/teacher/course/create', methods=['GET', 'POST'])
@login_required
@teacher_required
def teacher_create_course():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        level = request.form.get('level')
        duration_weeks = request.form.get('duration_weeks')
        topic = request.form.get('topic')
        
        course = Course(
            title=title,
            description=description,
            level=level,
            creator_id=current_user.id,
            duration_weeks=duration_weeks,
            topic=topic
        )
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"course_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                course.image = filename
        
        db.session.add(course)
        db.session.commit()
        
        flash('Khóa học đã được tạo thành công!', 'success')
        return redirect(url_for('teacher_courses'))
    
    return render_template('teacher/create_course.html')

@app.route('/teacher/course/<int:course_id>/edit', methods=['GET', 'POST'])
@login_required
@teacher_required
def teacher_edit_course(course_id):
    course = Course.query.get_or_404(course_id)
    
    # Kiểm tra quyền sở hữu
    if course.creator_id != current_user.id:
        flash('Bạn không có quyền chỉnh sửa khóa học này.', 'danger')
        return redirect(url_for('teacher_courses'))
    
    if request.method == 'POST':
        course.title = request.form.get('title')
        course.description = request.form.get('description')
        course.level = request.form.get('level')
        course.duration_weeks = request.form.get('duration_weeks')
        course.topic = request.form.get('topic')
        course.is_published = True if request.form.get('is_published') else False
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                filename = secure_filename(f"course_{int(datetime.now().timestamp())}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                course.image = filename
        
        db.session.commit()
        flash('Khóa học đã được cập nhật thành công!', 'success')
        return redirect(url_for('teacher_courses'))
    
    return render_template('teacher/edit_course.html', course=course)

@app.route('/teacher/course/<int:course_id>/delete', methods=['POST'])
@login_required
@teacher_required
def teacher_delete_course(course_id):
    course = Course.query.get_or_404(course_id)
    
    # Kiểm tra quyền sở hữu
    if course.creator_id != current_user.id:
        flash('Bạn không có quyền xóa khóa học này.', 'danger')
        return redirect(url_for('teacher_courses'))
    
    db.session.delete(course)
    db.session.commit()
    
    flash('Khóa học đã được xóa thành công!', 'success')
    return redirect(url_for('teacher_courses'))

@app.route('/teacher/course/<int:course_id>/lessons')
@login_required
@teacher_required
def teacher_course_lessons(course_id):
    course = Course.query.get_or_404(course_id)
    
    # Kiểm tra quyền sở hữu
    if course.creator_id != current_user.id:
        flash('Bạn không có quyền truy cập các bài học của khóa học này.', 'danger')
        return redirect(url_for('teacher_courses'))
    
    lessons = Lesson.query.filter_by(course_id=course.id).order_by(Lesson.order).all()
    
    return render_template('teacher/course_lessons.html', course=course, lessons=lessons)

@app.route('/teacher/course/<int:course_id>/students')
@login_required
@teacher_required
def teacher_course_students(course_id):
    course = Course.query.get_or_404(course_id)
    
    # Kiểm tra quyền sở hữu
    if course.creator_id != current_user.id:
        flash('Bạn không có quyền xem danh sách học sinh của khóa học này.', 'danger')
        return redirect(url_for('teacher_courses'))
    
    # Lấy danh sách học sinh đã đăng ký
    enrollments = CourseEnrollment.query.filter_by(course_id=course.id).all()
    students = [User.query.get(e.user_id) for e in enrollments]
    
    return render_template('teacher/course_students.html', course=course, students=students, enrollments=enrollments)

@app.route('/teacher/course/<int:course_id>/student/<int:user_id>')
@login_required
@teacher_required
def teacher_student_progress(course_id, user_id):
    course = Course.query.get_or_404(course_id)
    student = User.query.get_or_404(user_id)
    
    # Kiểm tra quyền sở hữu
    if course.creator_id != current_user.id:
        flash('Bạn không có quyền xem tiến độ học sinh trong khóa học này.', 'danger')
        return redirect(url_for('teacher_courses'))
    
    # Lấy thông tin đăng ký khóa học
    enrollment = CourseEnrollment.query.filter_by(user_id=user_id, course_id=course_id).first_or_404()
    
    # Lấy kết quả bài tập
    exercise_results = db.session.query(
        Exercise, ExerciseSubmission
    ).join(
        ExerciseSubmission, Exercise.id == ExerciseSubmission.exercise_id
    ).join(
        Lesson, Exercise.lesson_id == Lesson.id
    ).filter(
        Lesson.course_id == course_id,
        ExerciseSubmission.user_id == user_id
    ).all()
    
    return render_template('teacher/student_progress.html', 
                          course=course, 
                          student=student, 
                          enrollment=enrollment,
                          exercise_results=exercise_results)

@app.route('/teacher/tests')
@login_required
@teacher_required
def teacher_tests():
    # Các bài test do giáo viên tạo
    tests = Test.query.filter_by(creator_id=current_user.id).all()
    return render_template('teacher/tests.html', tests=tests)

@app.route('/teacher/test/create', methods=['GET', 'POST'])
@login_required
@teacher_required
def teacher_create_test():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        test_type = request.form.get('type')
        level = request.form.get('level') if request.form.get('type') != 'placement' else None
        time_limit = request.form.get('time_limit')
        passing_score = request.form.get('passing_score')
        
        # Tạo nội dung bài test từ AI
        if test_type == 'placement':
            ai_content = ai.generate_test()
        else:
            ai_content = ai.generate_test(level, None, int(request.form.get('num_questions', 10)))
        
        test = Test(
            title=title,
            description=description,
            type=test_type,
            creator_id=current_user.id,
            level=level,
            questions=json.dumps(ai_content["questions"]),
            time_limit=time_limit,
            passing_score=passing_score
        )
        db.session.add(test)
        db.session.commit()
        
        flash('Bài test đã được tạo thành công!', 'success')
        return redirect(url_for('teacher_tests'))
    
    return render_template('teacher/create_test.html')

@app.route('/teacher/test/<int:test_id>/results')
@login_required
@teacher_required
def teacher_test_results(test_id):
    test = Test.query.get_or_404(test_id)
    
    # Kiểm tra quyền sở hữu
    if test.creator_id != current_user.id:
        flash('Bạn không có quyền xem kết quả của bài test này.', 'danger')
        return redirect(url_for('teacher_tests'))
    
    # Lấy tất cả kết quả bài test
    results = TestResult.query.filter_by(test_id=test.id).order_by(TestResult.completed_at.desc()).all()
    
    # Lấy thông tin người dùng
    users = {r.user_id: User.query.get(r.user_id) for r in results}
    
    return render_template('teacher/test_results.html', test=test, results=results, users=users)

@app.route('/teacher/test/<int:test_id>/result/<int:result_id>')
@login_required
@teacher_required
def teacher_test_result_detail(test_id, result_id):
    test = Test.query.get_or_404(test_id)
    result = TestResult.query.get_or_404(result_id)
    
    # Kiểm tra quyền sở hữu
    if test.creator_id != current_user.id:
        flash('Bạn không có quyền xem chi tiết kết quả này.', 'danger')
        return redirect(url_for('teacher_tests'))
    
    # Lấy thông tin người dùng
    user = User.query.get(result.user_id)
    
    # Parse dữ liệu JSON
    feedback = json.loads(result.feedback)
    answers = json.loads(result.answers)
    questions = json.loads(test.questions)
    
    return render_template('teacher/test_result_detail.html', 
                          test=test, 
                          result=result, 
                          user=user,
                          feedback=feedback,
                          answers=answers,
                          questions=questions)

@app.route('/teacher/forum')
@login_required
@teacher_required
def teacher_forum():
    # Lấy các bài viết mới nhất
    recent_posts = ForumPost.query.order_by(ForumPost.created_at.desc()).limit(10).all()
    
    # Lấy các bài viết chưa có phản hồi
    unanswered_posts = db.session.query(ForumPost).outerjoin(
        ForumComment, ForumPost.id == ForumComment.post_id
    ).group_by(ForumPost.id).having(
        db.func.count(ForumComment.id) == 0
    ).order_by(ForumPost.created_at.desc()).limit(5).all()
    
    return render_template('teacher/forum.html', recent_posts=recent_posts, unanswered_posts=unanswered_posts)

# Hàm utility
def check_and_award_badges(user_id):
    """Kiểm tra và trao huy hiệu dựa trên thành tích của người dùng"""
    user_profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not user_profile:
        return
    
    # Lấy tất cả huy hiệu
    badges = Badge.query.all()
    
    # Lấy huy hiệu hiện tại của người dùng
    user_badges = UserBadge.query.filter_by(user_id=user_id).all()
    user_badge_ids = [ub.badge_id for ub in user_badges]
    
    awarded_badges = []
    
    for badge in badges:
        # Bỏ qua nếu đã có huy hiệu
        if badge.id in user_badge_ids:
            continue
        
        # Kiểm tra điều kiện dựa trên loại yêu cầu
        if badge.requirement_type == 'points' and user_profile.total_points >= badge.requirement_value:
            # Trao huy hiệu điểm
            new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
            db.session.add(new_badge)
            awarded_badges.append(badge)
            
        elif badge.requirement_type == 'streak' and user_profile.study_streak >= badge.requirement_value:
            # Trao huy hiệu chuỗi ngày học liên tiếp
            new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
            db.session.add(new_badge)
            awarded_badges.append(badge)
            
        elif badge.requirement_type == 'tests':
            # Đếm số bài test đã hoàn thành
            test_count = TestResult.query.filter_by(user_id=user_id).count()
            if test_count >= badge.requirement_value:
                new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(new_badge)
                awarded_badges.append(badge)
                
        elif badge.requirement_type == 'courses':
            # Đếm số khóa học đã hoàn thành
            completed_courses = CourseEnrollment.query.filter_by(
                user_id=user_id, status='completed'
            ).count()
            if completed_courses >= badge.requirement_value:
                new_badge = UserBadge(user_id=user_id, badge_id=badge.id)
                db.session.add(new_badge)
                awarded_badges.append(badge)
    
    # Lưu thay đổi nếu có huy hiệu mới
    if awarded_badges:
        db.session.commit()
        
        # Tạo thông báo về huy hiệu mới
        for badge in awarded_badges:
            notification = Notification(
                user_id=user_id,
                title=f"Chúc mừng! Bạn đã nhận được huy hiệu mới",
                content=f"Bạn vừa nhận được huy hiệu '{badge.name}'. {badge.description}",
                type="badge",
                url=url_for('badges')
            )
            db.session.add(notification)
        
        db.session.commit()
    
    return awarded_badges

def update_study_streak(user_id):
    """Cập nhật chuỗi ngày học liên tiếp"""
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        return
    
    today = datetime.utcnow().date()
    
    if not profile.last_study_date:
        # Lần đầu tiên học
        profile.study_streak = 1
        profile.last_study_date = today
    elif profile.last_study_date == today:
        # Đã cập nhật hôm nay rồi
        pass
    elif profile.last_study_date == today - timedelta(days=1):
        # Học liên tiếp
        profile.study_streak += 1
        profile.last_study_date = today
    else:
        # Bị gián đoạn
        profile.study_streak = 1
        profile.last_study_date = today
    
    db.session.commit()
    
    # Kiểm tra huy hiệu
    check_and_award_badges(user_id)

@app.before_request
def update_user_activity():
    """Cập nhật hoạt động của người dùng trước mỗi request"""
    if current_user.is_authenticated:
        # Kiểm tra xem người dùng đã học trong ngày chưa
        today = datetime.utcnow().date()
        
        # Tìm thống kê của ngày hôm nay
        stat = Statistic.query.filter_by(user_id=current_user.id, date=today).first()
        
        # Tạo thống kê mới nếu chưa có
        if not stat:
            stat = Statistic(user_id=current_user.id, date=today)
            db.session.add(stat)
            db.session.commit()
            
            # Cập nhật chuỗi ngày học liên tiếp
            update_study_streak(current_user.id)

# Hàm khởi tạo dữ liệu mẫu
def create_sample_data():
    """Tạo dữ liệu mẫu cho ứng dụng"""
    # Kiểm tra xem đã có dữ liệu chưa
    if User.query.first():
        return
    
    # Tạo người dùng admin
    admin_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
    admin = User(
        username='admin',
        email='admin@example.com',
        password=admin_password,
        fullname='Admin User',
        role='admin'
    )
    db.session.add(admin)
    
    # Tạo người dùng giáo viên
    teacher_password = bcrypt.generate_password_hash('teacher123').decode('utf-8')
    teacher = User(
        username='teacher',
        email='teacher@example.com',
        password=teacher_password,
        fullname='Teacher User',
        role='teacher'
    )
    db.session.add(teacher)
    
    # Tạo người dùng học sinh
    student_password = bcrypt.generate_password_hash('student123').decode('utf-8')
    student = User(
        username='student',
        email='student@example.com',
        password=student_password,
        fullname='Student User',
        role='student'
    )
    db.session.add(student)
    
    db.session.commit()
    
    # Tạo hồ sơ người dùng
    admin_profile = UserProfile(user_id=admin.id)
    teacher_profile = UserProfile(user_id=teacher.id)
    student_profile = UserProfile(
        user_id=student.id,
        language_level="Beginner (A1)",
        learning_goals="Tôi muốn học tiếng Anh để du lịch và công việc.",
        preferred_topics="Travel,Work,Culture"
    )
    
    db.session.add_all([admin_profile, teacher_profile, student_profile])
    
    # Tạo danh mục diễn đàn
    forum_categories = [
        ForumCategory(name="Thảo luận chung", description="Thảo luận về mọi chủ đề liên quan đến việc học tiếng Anh", order=1),
        ForumCategory(name="Ngữ pháp", description="Hỏi đáp về ngữ pháp tiếng Anh", order=2),
        ForumCategory(name="Từ vựng", description="Thảo luận về từ vựng và cách sử dụng", order=3),
        ForumCategory(name="Luyện thi", description="Thảo luận về các kỳ thi IELTS, TOEFL, TOEIC...", order=4)
    ]
    db.session.add_all(forum_categories)
    
    # Tạo huy hiệu
    badges = [
        Badge(name="Người mới", description="Hoàn thành bài test xếp loại đầu tiên", requirement_type="tests", requirement_value=1, image="badge_newcomer.png"),
        Badge(name="Siêng năng", description="Học liên tục 7 ngày", requirement_type="streak", requirement_value=7, image="badge_diligent.png"),
        Badge(name="Học giả", description="Đạt 1000 điểm", requirement_type="points", requirement_value=1000, image="badge_scholar.png"),
        Badge(name="Chuyên gia", description="Hoàn thành 5 khóa học", requirement_type="courses", requirement_value=5, image="badge_expert.png")
    ]
    db.session.add_all(badges)
    
    # Tạo phòng chat
    chat_rooms = [
        ChatRoom(name="Lớp học chung", description="Phòng chat chung cho tất cả học viên", is_private=False, creator_id=admin.id),
        ChatRoom(name="Hỏi đáp nhanh", description="Đặt câu hỏi và nhận câu trả lời nhanh chóng", is_private=False, creator_id=teacher.id)
    ]
    db.session.add_all(chat_rooms)
    
    # Tạo khóa học mẫu
    beginner_course = Course(
        title="Tiếng Anh cơ bản cho người mới bắt đầu",
        description="Khóa học này giúp bạn xây dựng nền tảng tiếng Anh vững chắc từ đầu, phù hợp cho người chưa có kiến thức nền.",
        level="Beginner (A1)",
        creator_id=teacher.id,
        is_published=True,
        duration_weeks=8,
        topic="Basic English",
        image="default_course.jpg"
    )
    
    intermediate_course = Course(
        title="Giao tiếp tiếng Anh thương mại",
        description="Khóa học tập trung vào các tình huống giao tiếp tiếng Anh trong môi trường kinh doanh và công sở.",
        level="Intermediate (B1)",
        creator_id=teacher.id,
        is_published=True,
        duration_weeks=10,
        topic="Business English",
        image="default_course.jpg"
    )
    
    db.session.add_all([beginner_course, intermediate_course])
    db.session.commit()
    
    # Tạo bài học mẫu
    beginner_lessons = [
        Lesson(
            course_id=beginner_course.id,
            title="Chào hỏi và giới thiệu bản thân",
            content=json.dumps(ai.generate_lesson("Beginner (A1)", "Greetings")),
            order=1,
            estimated_time=30
        ),
        Lesson(
            course_id=beginner_course.id,
            title="Các động từ cơ bản và thì hiện tại đơn",
            content=json.dumps(ai.generate_lesson("Beginner (A1)", "Basic Verbs")),
            order=2,
            estimated_time=45
        )
    ]
    db.session.add_all(beginner_lessons)
    
    # Tạo bài test mẫu
    placement_test = Test(
        title="Bài test xếp loại",
        description="Bài test này sẽ giúp xác định trình độ tiếng Anh của bạn.",
        type="placement",
        creator_id=admin.id,
        questions=json.dumps(ai.generate_test()["questions"]),
        time_limit=30,
        passing_score=7.0
    )
    
    beginner_test = Test(
        title="Kiểm tra trình độ A1",
        description="Bài test đánh giá kiến thức tiếng Anh cơ bản.",
        type="level",
        creator_id=teacher.id,
        level="Beginner (A1)",
        questions=json.dumps(ai.generate_test("Beginner (A1)")["questions"]),
        time_limit=20,
        passing_score=6.0
    )
    
    db.session.add_all([placement_test, beginner_test])
    db.session.commit()
    
    # Tạo bài tập mẫu cho bài học đầu tiên
    exercises = [
        Exercise(
            lesson_id=beginner_lessons[0].id,
            title="Bài tập về lời chào và giới thiệu",
            description="Thực hành các cách chào hỏi và giới thiệu bản thân trong tiếng Anh.",
            type="quiz",
            content=json.dumps(ai.generate_test("Beginner (A1)", "Greetings", 5)["questions"]),
            time_limit=10,
            points=10
        ),
        Exercise(
            lesson_id=beginner_lessons[0].id,
            title="Luyện nghe hội thoại chào hỏi",
            description="Nghe và trả lời câu hỏi về các hội thoại chào hỏi.",
            type="listening",
            content=json.dumps(ai.generate_test("Beginner (A1)", "Listening", 3)["questions"]),
            time_limit=15,
            points=15
        )
    ]
    db.session.add_all(exercises)
    
    # Đăng ký khóa học mẫu cho học sinh
    enrollment = CourseEnrollment(
        user_id=student.id,
        course_id=beginner_course.id,
        last_accessed=datetime.utcnow()
    )
    db.session.add(enrollment)
    
    # Tạo bài viết diễn đàn mẫu
    forum_post = ForumPost(
        category_id=forum_categories[0].id,
        user_id=student.id,
        title="Làm thế nào để học từ vựng hiệu quả?",
        content="Chào mọi người, mình đang gặp khó khăn trong việc nhớ từ vựng tiếng Anh. Mọi người có phương pháp nào hiệu quả không ạ? Xin cảm ơn!"
    )
    db.session.add(forum_post)
    
    # Tạo bình luận mẫu
    forum_comment = ForumComment(
        post_id=forum_post.id,
        user_id=teacher.id,
        content="Chào bạn, mình khuyên bạn nên kết hợp học từ vựng với hình ảnh và ngữ cảnh thực tế. Cách này giúp mình nhớ từ vựng lâu hơn nhiều."
    )
    db.session.add(forum_comment)
    
    # Tạo thông báo mẫu
    notification = Notification(
        user_id=student.id,
        title="Chào mừng đến với hệ thống học tiếng Anh trực tuyến!",
        content="Chúng tôi rất vui khi bạn tham gia. Hãy bắt đầu với bài test xếp loại để chúng tôi có thể gợi ý các khóa học phù hợp với trình độ của bạn.",
        type="system",
        url=url_for('placement_test')
    )
    db.session.add(notification)
    
    db.session.commit()
    
    print("Đã tạo dữ liệu mẫu thành công!")

# Tạo cơ sở dữ liệu và khởi chạy ứng dụng
@app.before_first_request
def initialize_db():
    """Tạo cơ sở dữ liệu và dữ liệu mẫu trước request đầu tiên"""
    print("Initializing database...")
    db.create_all()
    
    # Kiểm tra thư mục mô hình Deepseek tồn tại
    if os.path.exists('./deepseek-model'):
        print("Deepseek model directory found")
        model_files = os.listdir('./deepseek-model')
        print(f"Files in model directory: {', '.join(model_files[:5])}...")
    else:
        print("WARNING: Deepseek model directory not found")
        print("AI will run in simulation mode")
    
    create_sample_data()
    print("Sample data created successfully")

# Handler cho các lỗi 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Handler cho các lỗi 500
@app.errorhandler(500)
def inject_year():
    return {'current_year': datetime.now().year}

# Thêm biến toàn cục cho templates
@app.context_processor
def inject_global_vars():
    """Thêm các biến toàn cục cho templates"""
    def get_unread_notifications_count():
        if current_user.is_authenticated:
            return Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
        return 0
    
    return {
        'get_unread_notifications_count': get_unread_notifications_count
    }

# Chạy ứng dụng
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)