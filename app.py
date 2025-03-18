from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, g, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_mail import Mail, Message
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
import time
import schedule
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import wave
import struct
from PIL import Image, ImageDraw, ImageFont

print("Khởi động ứng dụng học tiếng Anh thông minh...")
print("Kiểm tra thư viện cần thiết...")

# Kiểm tra thư viện cần thiết
try:
    import torch
    print(f"Phiên bản PyTorch: {torch.__version__}")
    
    import transformers
    print(f"Phiên bản Transformers: {transformers.__version__}")
    
    if torch.cuda.is_available():
        print(f"GPU khả dụng: {torch.cuda.get_device_name(0)}")
    else:
        print("Không phát hiện GPU. Sử dụng CPU cho mô hình.")
        
    # Kiểm tra Accelerate
    try:
        import accelerate
        print(f"Phiên bản Accelerate: {accelerate.__version__}")
    except ImportError:
        print("CẢNH BÁO: Không tìm thấy thư viện Accelerate!")
        print("Cài đặt bằng lệnh: pip install 'accelerate>=0.26.0'")
    
    # Kiểm tra thư viện gTTS cho chuyển văn bản thành giọng nói
    try:
        from gtts import gTTS
        print("Đã tìm thấy gTTS cho tạo audio")
    except ImportError:
        print("CẢNH BÁO: Không tìm thấy thư viện gTTS cho audio!")
        print("Đang tự động cài đặt gTTS...")
        os.system('pip install gtts')
        from gtts import gTTS
        print("Đã cài đặt thành công gTTS")
        
    # Kiểm tra Pillow cho xử lý hình ảnh
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("Đã tìm thấy Pillow cho xử lý hình ảnh")
    except ImportError:
        print("CẢNH BÁO: Không tìm thấy thư viện Pillow cho xử lý hình ảnh!")
        print("Đang tự động cài đặt Pillow...")
        os.system('pip install pillow')
        from PIL import Image, ImageDraw, ImageFont
        print("Đã cài đặt thành công Pillow")
    
    # Kiểm tra SpeechRecognition cho nhận dạng giọng nói
    try:
        import speech_recognition as sr
        print("Đã tìm thấy SpeechRecognition cho nhận dạng giọng nói")
    except ImportError:
        print("CẢNH BÁO: Không tìm thấy thư viện SpeechRecognition!")
        print("Đang tự động cài đặt SpeechRecognition...")
        os.system('pip install SpeechRecognition')
        import speech_recognition as sr
        print("Đã cài đặt thành công SpeechRecognition")
    
    # Kiểm tra matplotlib cho tạo biểu đồ
    try:
        import matplotlib
        print(f"Đã tìm thấy matplotlib phiên bản {matplotlib.__version__}")
    except ImportError:
        print("CẢNH BÁO: Không tìm thấy thư viện matplotlib!")
        print("Đang tự động cài đặt matplotlib...")
        os.system('pip install matplotlib')
        import matplotlib
        print("Đã cài đặt thành công matplotlib")
    
    # Sử dụng backend non-interactive cho matplotlib
    matplotlib.use('Agg')
        
except ImportError as e:
    print(f"Thiếu thư viện cần thiết: {e}")
    print("Vui lòng cài đặt: pip install torch transformers")


# Lớp tiện ích xử lý media (audio, hình ảnh, video)
class TienIchMedia:
    """
    Lớp tiện ích cho việc tạo và xử lý các tệp media 
    phục vụ các bài học tiếng Anh bao gồm nói, nghe, đọc, viết
    """
    
    @staticmethod
    def tao_audio_tu_van_ban(van_ban, duong_dan, ngon_ngu='en'):
        """
        Tạo file audio từ văn bản sử dụng gTTS
        
        Tham số:
        van_ban (str): Văn bản cần chuyển thành giọng nói
        duong_dan (str): Đường dẫn lưu file audio
        ngon_ngu (str): Mã ngôn ngữ ('en', 'vi',...)
        
        Trả về:
        str: Đường dẫn đến file audio đã tạo
        """
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Tạo audio từ văn bản sử dụng gTTS
            tts = gTTS(text=van_ban, lang=ngon_ngu, slow=False)
            tts.save(duong_dan)
            
            print(f"Đã tạo audio thành công: {duong_dan}")
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo audio: {str(e)}")
            # Tạo file audio trống trong trường hợp lỗi
            return TienIchMedia.tao_audio_trang(duong_dan)
    
    @staticmethod
    def tao_audio_trang(duong_dan, thoi_gian=2):
        """
        Tạo file audio trống (im lặng) làm dự phòng
        
        Tham số:
        duong_dan (str): Đường dẫn lưu file audio
        thoi_gian (int): Thời gian im lặng (giây)
        
        Trả về:
        str: Đường dẫn đến file audio đã tạo
        """
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Tạo file WAV trống
            sample_rate = 44100
            audio = wave.open(duong_dan, 'w')
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            
            # Tạo dữ liệu audio trống
            for _ in range(int(thoi_gian * sample_rate)):
                value = 0
                packed_value = struct.pack('h', value)
                audio.writeframes(packed_value)
            
            audio.close()
            print(f"Đã tạo audio trống: {duong_dan}")
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo audio trống: {str(e)}")
            return None
    
    @staticmethod
    def tao_hinh_anh_van_ban(van_ban, duong_dan, chieu_rong=800, chieu_cao=400, 
                             tieu_de="Bài tập đọc hiểu", mau_nen=(255, 255, 255)):
        """
        Tạo hình ảnh chứa văn bản cho bài đọc
        
        Tham số:
        van_ban (str): Nội dung văn bản
        duong_dan (str): Đường dẫn lưu hình ảnh
        chieu_rong (int): Chiều rộng hình ảnh
        chieu_cao (int): Chiều cao hình ảnh
        tieu_de (str): Tiêu đề trên hình ảnh
        mau_nen (tuple): Màu nền (R,G,B)
        
        Trả về:
        str: Đường dẫn đến hình ảnh đã tạo
        """
        try:
            # Tạo hình ảnh trống với nền trắng
            hinh_anh = Image.new('RGB', (chieu_rong, chieu_cao), color=mau_nen)
            ve = ImageDraw.Draw(hinh_anh)
            
            # Thiết lập font (sử dụng font mặc định nếu không tìm thấy font chỉ định)
            try:
                font_tieu_de = ImageFont.truetype("arial.ttf", 24)
                font_noi_dung = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font_tieu_de = ImageFont.load_default()
                font_noi_dung = ImageFont.load_default()
            
            # Vẽ viền
            ve.rectangle([(0, 0), (chieu_rong-1, chieu_cao-1)], outline=(200, 200, 200))
            
            # Vẽ tiêu đề
            do_rong_tieu_de = ve.textlength(tieu_de, font=font_tieu_de)
            ve.text(((chieu_rong - do_rong_tieu_de) // 2, 20), tieu_de, fill=(0, 0, 0), font=font_tieu_de)
            
            # Vẽ văn bản (chia thành nhiều dòng để vừa với chiều rộng)
            bien = 40
            vi_tri_y = 70
            max_chieu_rong = chieu_rong - 2 * bien
            
            # Chia văn bản thành các từ
            tu = van_ban.split()
            dong = []
            dong_hien_tai = []
            
            for tu_don in tu:
                dong_thu = ' '.join(dong_hien_tai + [tu_don])
                
                # Kiểm tra chiều rộng của dòng hiện tại
                do_rong_dong = ve.textlength(dong_thu, font=font_noi_dung)
                
                if do_rong_dong <= max_chieu_rong:
                    dong_hien_tai.append(tu_don)
                else:
                    dong.append(' '.join(dong_hien_tai))
                    dong_hien_tai = [tu_don]
            
            # Thêm dòng cuối cùng
            if dong_hien_tai:
                dong.append(' '.join(dong_hien_tai))
            
            # Vẽ các dòng văn bản
            for dong_van_ban in dong:
                ve.text((bien, vi_tri_y), dong_van_ban, fill=(0, 0, 0), font=font_noi_dung)
                vi_tri_y += 25
                
                # Kiểm tra nếu vượt quá chiều cao hình ảnh
                if vi_tri_y > chieu_cao - bien:
                    break
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Lưu hình ảnh
            hinh_anh.save(duong_dan)
            print(f"Đã tạo hình ảnh văn bản: {duong_dan}")
            
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo hình ảnh văn bản: {str(e)}")
            return TienIchMedia.tao_hinh_anh_macdinh(duong_dan)
    
    @staticmethod
    def tao_hinh_anh_tu_vung(tu_vung_list, duong_dan, chieu_rong=800, chieu_cao=600):
        """
        Tạo hình ảnh danh sách từ vựng cho bài học
        
        Tham số:
        tu_vung_list (list): Danh sách từ vựng dạng [{"word": "...", "meaning": "...", "example": "..."}]
        duong_dan (str): Đường dẫn lưu hình ảnh
        chieu_rong (int): Chiều rộng hình ảnh
        chieu_cao (int): Chiều cao hình ảnh
        
        Trả về:
        str: Đường dẫn đến hình ảnh đã tạo
        """
        try:
            # Tạo hình ảnh trống với nền trắng
            hinh_anh = Image.new('RGB', (chieu_rong, chieu_cao), color=(255, 255, 255))
            ve = ImageDraw.Draw(hinh_anh)
            
            # Thiết lập font
            try:
                font_tieu_de = ImageFont.truetype("arial.ttf", 24)
                font_tu = ImageFont.truetype("arial.ttf", 18)
                font_nghia = ImageFont.truetype("arial.ttf", 16)
                font_vi_du = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font_tieu_de = ImageFont.load_default()
                font_tu = ImageFont.load_default()
                font_nghia = ImageFont.load_default()
                font_vi_du = ImageFont.load_default()
            
            # Vẽ viền
            ve.rectangle([(0, 0), (chieu_rong-1, chieu_cao-1)], outline=(200, 200, 200))
            
            # Vẽ tiêu đề
            tieu_de = "Từ vựng tiếng Anh"
            do_rong_tieu_de = ve.textlength(tieu_de, font=font_tieu_de)
            ve.text(((chieu_rong - do_rong_tieu_de) // 2, 20), tieu_de, fill=(0, 0, 100), font=font_tieu_de)
            
            # Vẽ danh sách từ vựng
            vi_tri_y = 70
            bien = 40
            
            for i, tu in enumerate(tu_vung_list[:10]):  # Giới hạn 10 từ để tránh quá tải
                # Vẽ từ
                ve.text((bien, vi_tri_y), f"{i+1}. {tu['word']}", fill=(0, 0, 0), font=font_tu)
                vi_tri_y += 30
                
                # Vẽ nghĩa
                ve.text((bien + 20, vi_tri_y), f"Nghĩa: {tu['meaning']}", fill=(100, 0, 0), font=font_nghia)
                vi_tri_y += 25
                
                # Vẽ ví dụ
                if 'example' in tu and tu['example']:
                    # Chia ví dụ thành nhiều dòng nếu cần
                    vi_du = tu['example']
                    max_chieu_rong = chieu_rong - 2 * (bien + 20)
                    
                    # Chia ví dụ thành các từ
                    tu_vi_du = vi_du.split()
                    dong_vi_du = []
                    dong_hien_tai = []
                    
                    for tu_don in tu_vi_du:
                        dong_thu = ' '.join(dong_hien_tai + [tu_don])
                        
                        # Kiểm tra chiều rộng
                        do_rong_dong = ve.textlength(dong_thu, font=font_vi_du)
                        
                        if do_rong_dong <= max_chieu_rong:
                            dong_hien_tai.append(tu_don)
                        else:
                            dong_vi_du.append(' '.join(dong_hien_tai))
                            dong_hien_tai = [tu_don]
                    
                    # Thêm dòng cuối
                    if dong_hien_tai:
                        dong_vi_du.append(' '.join(dong_hien_tai))
                    
                    # Vẽ các dòng ví dụ
                    ve.text((bien + 20, vi_tri_y), "Ví dụ:", fill=(0, 100, 0), font=font_vi_du)
                    vi_tri_y += 20
                    
                    for dong in dong_vi_du:
                        ve.text((bien + 40, vi_tri_y), dong, fill=(0, 100, 0), font=font_vi_du)
                        vi_tri_y += 20
                
                vi_tri_y += 20
                
                # Kiểm tra nếu vượt quá chiều cao
                if vi_tri_y > chieu_cao - bien:
                    break
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Lưu hình ảnh
            hinh_anh.save(duong_dan)
            print(f"Đã tạo hình ảnh từ vựng: {duong_dan}")
            
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo hình ảnh từ vựng: {str(e)}")
            return TienIchMedia.tao_hinh_anh_macdinh(duong_dan, noi_dung="Danh sách từ vựng")
    
    @staticmethod
    def tao_hinh_anh_ngu_phap(ngu_phap_items, duong_dan, chieu_rong=800, chieu_cao=600):
        """
        Tạo hình ảnh minh họa ngữ pháp tiếng Anh
        
        Tham số:
        ngu_phap_items (list): Danh sách cấu trúc ngữ pháp dạng [{"title": "...", "explanation": "...", "examples": [...]}]
        duong_dan (str): Đường dẫn lưu hình ảnh
        chieu_rong (int): Chiều rộng hình ảnh
        chieu_cao (int): Chiều cao hình ảnh
        
        Trả về:
        str: Đường dẫn đến hình ảnh đã tạo
        """
        try:
            # Tạo hình ảnh trống với nền trắng
            hinh_anh = Image.new('RGB', (chieu_rong, chieu_cao), color=(255, 255, 255))
            ve = ImageDraw.Draw(hinh_anh)
            
            # Thiết lập font
            try:
                font_tieu_de = ImageFont.truetype("arial.ttf", 24)
                font_cau_truc = ImageFont.truetype("arial.ttf", 18)
                font_giai_thich = ImageFont.truetype("arial.ttf", 16)
                font_vi_du = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font_tieu_de = ImageFont.load_default()
                font_cau_truc = ImageFont.load_default()
                font_giai_thich = ImageFont.load_default()
                font_vi_du = ImageFont.load_default()
            
            # Vẽ viền
            ve.rectangle([(0, 0), (chieu_rong-1, chieu_cao-1)], outline=(200, 200, 200))
            
            # Vẽ tiêu đề
            tieu_de = "Ngữ pháp tiếng Anh"
            do_rong_tieu_de = ve.textlength(tieu_de, font=font_tieu_de)
            ve.text(((chieu_rong - do_rong_tieu_de) // 2, 20), tieu_de, fill=(0, 0, 100), font=font_tieu_de)
            
            # Vẽ các cấu trúc ngữ pháp
            vi_tri_y = 70
            bien = 40
            
            for i, item in enumerate(ngu_phap_items):
                # Vẽ tiêu đề cấu trúc
                ve.text((bien, vi_tri_y), f"{i+1}. {item['title']}", fill=(0, 0, 0), font=font_cau_truc)
                vi_tri_y += 30
                
                # Vẽ giải thích
                max_chieu_rong = chieu_rong - 2 * (bien + 20)
                giai_thich = item['explanation']
                
                # Chia giải thích thành các từ
                tu_giai_thich = giai_thich.split()
                dong_giai_thich = []
                dong_hien_tai = []
                
                for tu_don in tu_giai_thich:
                    dong_thu = ' '.join(dong_hien_tai + [tu_don])
                    
                    # Kiểm tra chiều rộng
                    do_rong_dong = ve.textlength(dong_thu, font=font_giai_thich)
                    
                    if do_rong_dong <= max_chieu_rong:
                        dong_hien_tai.append(tu_don)
                    else:
                        dong_giai_thich.append(' '.join(dong_hien_tai))
                        dong_hien_tai = [tu_don]
                
                # Thêm dòng cuối
                if dong_hien_tai:
                    dong_giai_thich.append(' '.join(dong_hien_tai))
                
                # Vẽ các dòng giải thích
                for dong in dong_giai_thich:
                    ve.text((bien + 20, vi_tri_y), dong, fill=(0, 0, 0), font=font_giai_thich)
                    vi_tri_y += 25
                
                vi_tri_y += 10
                
                # Vẽ ví dụ
                if 'examples' in item and item['examples']:
                    ve.text((bien + 20, vi_tri_y), "Ví dụ:", fill=(100, 0, 0), font=font_vi_du)
                    vi_tri_y += 20
                    
                    for j, vi_du in enumerate(item['examples']):
                        ve.text((bien + 40, vi_tri_y), f"- {vi_du}", fill=(100, 0, 0), font=font_vi_du)
                        vi_tri_y += 20
                
                vi_tri_y += 20
                
                # Kiểm tra nếu vượt quá chiều cao
                if vi_tri_y > chieu_cao - bien:
                    break
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Lưu hình ảnh
            hinh_anh.save(duong_dan)
            print(f"Đã tạo hình ảnh ngữ pháp: {duong_dan}")
            
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo hình ảnh ngữ pháp: {str(e)}")
            return TienIchMedia.tao_hinh_anh_macdinh(duong_dan, noi_dung="Ngữ pháp tiếng Anh")
    
    @staticmethod
    def tao_hinh_anh_macdinh(duong_dan, chieu_rong=800, chieu_cao=400, noi_dung="Hình ảnh mặc định"):
        """
        Tạo hình ảnh mặc định khi có lỗi
        
        Tham số:
        duong_dan (str): Đường dẫn lưu hình ảnh
        chieu_rong (int): Chiều rộng hình ảnh
        chieu_cao (int): Chiều cao hình ảnh
        noi_dung (str): Nội dung hiển thị trên hình ảnh
        
        Trả về:
        str: Đường dẫn đến hình ảnh đã tạo
        """
        try:
            # Tạo hình ảnh trống với nền xám nhạt
            hinh_anh = Image.new('RGB', (chieu_rong, chieu_cao), color=(240, 240, 240))
            ve = ImageDraw.Draw(hinh_anh)
            
            # Thiết lập font (sử dụng font mặc định nếu không tìm thấy font chỉ định)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Vẽ viền
            ve.rectangle([(0, 0), (chieu_rong-1, chieu_cao-1)], outline=(180, 180, 180))
            
            # Tính toán vị trí văn bản để căn giữa
            do_rong_van_ban = ve.textlength(noi_dung, font=font)
            vi_tri_van_ban = ((chieu_rong - do_rong_van_ban) // 2, chieu_cao // 2 - 10)
            
            # Vẽ văn bản
            ve.text(vi_tri_van_ban, noi_dung, fill=(100, 100, 100), font=font)
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Lưu hình ảnh
            hinh_anh.save(duong_dan)
            print(f"Đã tạo hình ảnh mặc định: {duong_dan}")
            
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo hình ảnh mặc định: {str(e)}")
            return None
    
    @staticmethod
    def tao_bieu_do_tien_bo(du_lieu, duong_dan, tieu_de="Biểu đồ tiến độ học tập", 
                         chieu_rong=800, chieu_cao=400):
        """
        Tạo biểu đồ tiến độ học tập
        
        Tham số:
        du_lieu (dict): Dữ liệu dạng {"labels": [...], "values": [...]}
        duong_dan (str): Đường dẫn lưu biểu đồ
        tieu_de (str): Tiêu đề biểu đồ
        chieu_rong (int): Chiều rộng biểu đồ
        chieu_cao (int): Chiều cao biểu đồ
        
        Trả về:
        str: Đường dẫn đến biểu đồ đã tạo
        """
        try:
            # Tạo hình vẽ với kích thước chỉ định
            plt.figure(figsize=(chieu_rong/100, chieu_cao/100), dpi=100)
            
            # Vẽ biểu đồ đường
            plt.plot(du_lieu['labels'], du_lieu['values'], marker='o', linestyle='-', color='blue')
            
            # Thiết lập thông tin biểu đồ
            plt.title(tieu_de)
            plt.xlabel('Thời gian')
            plt.ylabel('Điểm số')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Đặt giới hạn trục y từ 0 đến 100
            plt.ylim([0, 100])
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
            
            # Lưu biểu đồ
            plt.savefig(duong_dan, bbox_inches='tight')
            plt.close()  # Đóng hình để giải phóng bộ nhớ
            
            print(f"Đã tạo biểu đồ tiến độ: {duong_dan}")
            return duong_dan
        except Exception as e:
            print(f"Lỗi khi tạo biểu đồ tiến độ: {str(e)}")
            return TienIchMedia.tao_hinh_anh_macdinh(duong_dan, noi_dung="Biểu đồ tiến độ học tập")

# Tích hợp với mô hình Deepseek offline
class DeepseekAI:
    def __init__(self):
        # Các cấp độ và chủ đề
        self.levels = ["Beginner (A1)", "Elementary (A2)", "Intermediate (B1)", 
                    "Upper Intermediate (B2)", "Advanced (C1)", "Proficient (C2)"]
        self.topics = ["Greetings", "Family", "Food", "Travel", "Work", "Hobbies", 
                    "Health", "Environment", "Technology", "Education", "Culture",
                    "Sports", "Entertainment", "Shopping", "Business", "Science",
                    "Art", "Music", "History", "Geography", "Politics", "Economics",
                    "Literature", "Philosophy", "Psychology", "Sociology"]
        
        # Thêm mới các chủ đề bài học theo kỹ năng
        self.skill_topics = {
            "listening": ["Nghe hiểu hội thoại", "Nghe tin tức", "Nghe bài giảng", "Nghe và ghi chú", 
                         "Nghe phân biệt âm", "Nghe và trả lời câu hỏi"],
            "speaking": ["Giao tiếp hàng ngày", "Thuyết trình", "Tranh luận", "Mô tả hình ảnh", 
                        "Kể chuyện", "Phỏng vấn", "Phát âm"],
            "reading": ["Đọc hiểu văn bản", "Đọc tin tức", "Đọc tài liệu học thuật", "Đọc và tóm tắt", 
                       "Đọc và phân tích", "Đọc văn học"],
            "writing": ["Viết email", "Viết luận", "Viết báo cáo", "Viết tóm tắt", "Viết sáng tạo", 
                       "Viết đơn xin việc", "Viết blog"]
        }
        
        # Chủ đề theo cấp độ
        self.level_topics = {
            "Beginner (A1)": ["Greetings", "Family", "Numbers", "Colors", "Daily Activities", 
                             "Time", "Days and Months", "Food", "Simple Conversations"],
            "Elementary (A2)": ["Travel", "Shopping", "Hobbies", "Weather", "Clothes", 
                              "Health", "School", "Work", "House and Home"],
            "Intermediate (B1)": ["Environment", "Technology", "Entertainment", "Sports", 
                                "Culture", "Health and Fitness", "Education", "Work Life"],
            "Upper Intermediate (B2)": ["Media", "Science", "Arts", "Business", "Current Affairs", 
                                      "Social Issues", "Psychology", "Travel and Tourism"],
            "Advanced (C1)": ["Literature", "Economics", "Politics", "Philosophy", 
                            "Academic Subjects", "Global Issues", "Professional Development"],
            "Proficient (C2)": ["Advanced Literature", "Research Topics", "Complex Debates", 
                              "Specialized Fields", "Critical Analysis", "Creative Writing"]
        }
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Đường dẫn đến thư mục chứa mô hình Deepseek
            self.model_path = "./deepseek-model"
            
            # Thông báo log
            print("Đang tải mô hình Deepseek từ", self.model_path)
            
            # Tải tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Kiểm tra GPU
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                print("Sử dụng GPU:", torch.cuda.get_device_name(0))
                torch_dtype = torch.float16
            else:
                print("Sử dụng CPU cho mô hình")
                torch_dtype = torch.float32
            
            # Kiểm tra thư viện Accelerate
            has_accelerate = False
            try:
                import accelerate
                has_accelerate = True
                print("Đã phát hiện thư viện Accelerate:", accelerate.__version__)
            except ImportError:
                print("Không tìm thấy thư viện Accelerate, sử dụng chế độ tải cơ bản")
            
            # Tải mô hình dựa vào có Accelerate hay không
            if has_accelerate:
                # Nếu có Accelerate, sử dụng cấu hình đầy đủ
                device_map = "auto" if has_gpu else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # Nếu không có Accelerate, tránh sử dụng các tham số yêu cầu Accelerate
                if has_gpu:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True
                    )
                    self.model = self.model.to("cuda")
                else:
                    # Thêm vào phần tải mô hình (CPU only) để tối ưu hóa
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            quantization_config=quantization_config,
                            trust_remote_code=True
                        )
                        print("Sử dụng lượng tử hóa 8-bit để tối ưu bộ nhớ")
                    except Exception as e:
                        print(f"Không thể sử dụng lượng tử hóa: {e}, sử dụng chế độ tải tiêu chuẩn")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True
                        )
                        self.model = self.model.to("cpu")
                
            # Đặt model vào chế độ evaluation
            self.model.eval()
            print("Đã tải mô hình thành công!")
            self.model_loaded = True
            
            # Khởi tạo cache và thời gian cho tinh năng "mỗi ngày thông minh hơn"
            self.response_cache = {}
            self.last_creative_update = datetime.now() - timedelta(days=1)
            self.creativity_level = 0.7  # Mức độ sáng tạo mặc định
            
        except Exception as e:
            print(f"Lỗi khi tải mô hình Deepseek: {str(e)}")
            print("Chuyển sang chế độ mô phỏng...")
            self.model_loaded = False
    
    def update_creativity_daily(self):
        """Cập nhật mức độ sáng tạo mỗi ngày để mô hình 'thông minh hơn mỗi ngày'"""
        current_time = datetime.now()
        # Nếu đã qua một ngày kể từ lần cập nhật cuối
        if (current_time - self.last_creative_update).days >= 1:
            # Tăng độ sáng tạo nhưng giới hạn ở mức 0.95 (để tránh quá nhiều hallucination)
            self.creativity_level = min(0.95, self.creativity_level + 0.01)
            self.last_creative_update = current_time
            print(f"Đã cập nhật mức độ sáng tạo AI: {self.creativity_level:.2f}")
            
            # Xóa cache mỗi khi cập nhật độ sáng tạo để có nội dung mới
            self.response_cache = {}
            return True
        return False
    
    def _generate_response(self, prompt, max_length=2000, temperature=None):
        """Tạo phản hồi từ mô hình Deepseek với cache để tối ưu hóa hiệu suất"""
        try:
            # Cập nhật mức độ sáng tạo hàng ngày
            self.update_creativity_daily()
            
            # Sử dụng mức độ sáng tạo hiện tại nếu không có chỉ định cụ thể
            if temperature is None:
                temperature = self.creativity_level
                
            if not self.model_loaded:
                raise Exception("Mô hình chưa được tải")
                
            # Kiểm tra cache
            if not hasattr(self, 'response_cache'):
                self.response_cache = {}
                
            # Tạo cache key từ prompt và tham số
            cache_key = f"{prompt[:100]}_{max_length}_{temperature}"
            
            # Trả về kết quả từ cache nếu có
            if cache_key in self.response_cache:
                print("Sử dụng kết quả từ cache")
                return self.response_cache[cache_key]
            
            import torch
            
            # Giới hạn độ dài prompt nếu quá lớn để tránh lỗi
            max_input_length = 4096  # Số token tối đa mà model có thể xử lý
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if input_ids.shape[1] > max_input_length:
                print(f"Cảnh báo: Prompt quá dài ({input_ids.shape[1]} tokens), cắt bớt còn {max_input_length} tokens")
                input_ids = input_ids[:, :max_input_length]
                prompt = self.tokenizer.decode(input_ids[0])
            
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
                try:
                    # Kiểm tra xem có đủ bộ nhớ không
                    if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        if free_mem < 1024 * 1024 * 1024:  # 1GB
                            print("Cảnh báo: Bộ nhớ GPU thấp, sử dụng cài đặt sinh văn bản tiết kiệm hơn")
                            max_length = min(max_length, 1000)
                    
                    # Tạo đầu ra với tham số phù hợp    
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError as e:
                    # Nếu hết bộ nhớ, thử lại với cài đặt bảo thủ hơn
                    if "CUDA out of memory" in str(e) or "DefaultCPUAllocator: can't allocate memory" in str(e):
                        print("Lỗi bộ nhớ, thử lại với cài đặt bảo thủ hơn")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=min(1000, max_length // 2),
                            temperature=temperature,
                            top_p=0.9,
                            top_k=50,
                            do_sample=True,
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise
            
            # Decode kết quả
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Tách phần text sinh ra so với prompt
            prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            result = full_text[len(prompt_text):]
            
            # Lưu vào cache
            self.response_cache[cache_key] = result
            
            # Giới hạn kích thước cache
            if len(self.response_cache) > 100:  # Giữ tối đa 100 câu trả lời trong cache
                # Xóa mục cũ nhất
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            
            return result
                
        except Exception as e:
            print(f"Lỗi khi tạo phản hồi: {str(e)}")
            # Thử lấy stack trace để debug
            import traceback
            traceback.print_exc()
            
            # Trả về phản hồi dự phòng trong trường hợp lỗi
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """Tạo phản hồi dự phòng khi mô hình gặp lỗi"""
        if "test" in prompt.lower() or "bài kiểm tra" in prompt.lower():
            return """
            {
              "questions": [
                {
                  "id": "q1",
                  "type": "multiple_choice",
                  "question": "What is the meaning of 'Hello'?",
                  "options": ["Tạm biệt", "Xin chào", "Cảm ơn", "Xin lỗi"],
                  "correct_answer": "Xin chào",
                  "difficulty": "Beginner (A1)"
                },
                {
                  "id": "q2",
                  "type": "multiple_choice",
                  "question": "How do you say 'Goodbye' in English?",
                  "options": ["Hello", "Thanks", "Sorry", "Goodbye"],
                  "correct_answer": "Goodbye",
                  "difficulty": "Beginner (A1)"
                }
              ],
              "time_limit": 600,
              "passing_score": 6.0
            }
            """
        elif "lesson" in prompt.lower() or "bài học" in prompt.lower():
            return """
            {
              "title": "Basic Greetings",
              "description": "Learn common English greetings and introductions",
              "objectives": ["Learn basic greetings", "Practice introductions", "Understand formal vs informal greetings"],
              "sections": [
                {
                  "title": "Common Greetings",
                  "content": "<p>Here are some common English greetings:</p><ul><li>Hello</li><li>Hi</li><li>Good morning</li><li>Good afternoon</li><li>Good evening</li></ul>",
                  "vocabulary_list": [
                    {"word": "hello", "meaning": "xin chào", "example": "Hello, how are you?"},
                    {"word": "morning", "meaning": "buổi sáng", "example": "Good morning, everyone!"}
                  ]
                }
              ]
            }
            """
        else:
            return "I'm sorry, I couldn't generate a proper response. Please try again with a different prompt."
    
    def _extract_json_from_text(self, text):
        """Trích xuất JSON từ phản hồi của mô hình"""
        if not text:
            return None
            
        try:
            # Tìm JSON trong văn bản
            import re
            json_pattern = r'```json\n(.*?)\n```'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Thử tìm trong dấu ngoặc nhọn
            json_pattern = r'(\{.*\})'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
                
            # Không tìm thấy JSON
            return None
            
        except Exception as e:
            print(f"Lỗi khi trích xuất JSON: {str(e)}")
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
Hãy đánh giá chi tiết các điểm mạnh, điểm yếu và đưa ra đề xuất cụ thể dựa trên kết quả bài làm. Nếu phát hiện điểm yếu về kỹ năng nghe, nói, đọc hoặc viết, hãy nêu rõ đề xuất cải thiện cho từng kỹ năng.
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
            "strengths": ["Từ vựng cơ bản" if score > 5 else "Khả năng hiểu câu đơn giản", 
                        "Ngữ pháp cơ bản" if score > 6 else "Khả năng ghi nhớ từ vựng"],
            "weaknesses": ["Ngữ pháp nâng cao" if score < 8 else "Từ vựng học thuật", 
                         "Kỹ năng nghe" if score < 7 else "Kỹ năng viết nâng cao"],
            "recommendations": [
                f"Nên tập trung vào {'ngữ pháp nâng cao' if score < 8 else 'từ vựng chuyên ngành'}",
                f"Thực hành {'nghe và nói' if score < 7 else 'đọc và viết'} thường xuyên",
                "Làm nhiều bài tập đọc hiểu để nâng cao vốn từ vựng",
                "Luyện nghe với các video có phụ đề tiếng Anh"
            ]
        }
        return feedback
    
    def generate_test(self, level=None, topic=None, length=10):
        """Tạo bài kiểm tra dựa trên cấp độ và chủ đề"""
        if self.model_loaded:
            # Xác định loại bài kiểm tra
            test_type = "placement" if not level else "level"
            
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy tạo một bài kiểm tra tiếng Anh đầy đủ với các thông số sau:

Loại bài kiểm tra: {test_type}
{"Cấp độ: " + level if level else "Bài kiểm tra xếp loại từ cơ bản đến nâng cao"}
{"Chủ đề: " + topic if topic else ""}
Số lượng câu hỏi: {length}

Bài kiểm tra cần bao gồm các kỹ năng tiếng Anh: nghe, nói, đọc, viết. Mỗi kỹ năng nên có câu hỏi phù hợp.

Vui lòng tạo bài kiểm tra ở định dạng JSON theo cấu trúc sau:
```json
{{
  "questions": [
    {{
      "id": "q1",
      "type": "multiple_choice/writing/listening/speaking/grammar/vocabulary/reading/cloze",
      "question": "Nội dung câu hỏi",
      "options": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
      "correct_answer": "Đáp án đúng (hoặc hướng dẫn chấm điểm cho câu hỏi viết)",
      "audio_text": "Văn bản để tạo file audio cho câu hỏi nghe (nếu có)",
      "difficulty": "Cấp độ của câu hỏi",
      "skill": "Kỹ năng (listening/speaking/reading/writing)",
      "points": 1
    }},
    ...
  ],
  "time_limit": "Thời gian làm bài tính bằng giây",
  "passing_score": "Điểm để đạt (từ 0-10)",
  "instructions": "Hướng dẫn làm bài"
}}
```

Lưu ý:
- Với bài kiểm tra xếp loại, tạo câu hỏi có độ khó tăng dần từ A1 đến C2
- Đa dạng loại câu hỏi: trắc nghiệm, điền từ, đọc hiểu, nghe hiểu, nói, viết, ngữ pháp và từ vựng
- Với câu hỏi nghe, hãy thêm trường "audio_text" chứa nội dung văn bản để tạo file audio
- Với câu hỏi nói và viết, hãy cung cấp hướng dẫn chấm điểm chi tiết trong trường "correct_answer"
- Đảm bảo bài kiểm tra đánh giá đầy đủ cả 4 kỹ năng: nghe, nói, đọc, viết
- Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo bài kiểm tra
            response = self._generate_response(prompt, max_length=4000)
            test_data = self._extract_json_from_text(response)
            
            if test_data and "questions" in test_data:
                # Đảm bảo các trường cần thiết tồn tại
                if "time_limit" not in test_data:
                    test_data["time_limit"] = 30 * length
                if "passing_score" not in test_data:
                    test_data["passing_score"] = 7.0
                if "instructions" not in test_data:
                    test_data["instructions"] = "Hãy làm bài kiểm tra trong thời gian quy định. Đọc kỹ yêu cầu trước khi trả lời."
                
                # Thêm ID nếu thiếu
                for i, question in enumerate(test_data["questions"]):
                    if "id" not in question:
                        question["id"] = f"q{i+1}"
                    if "points" not in question:
                        question["points"] = 1
                
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
                        "question": f"Hãy nghe đoạn hội thoại và chọn câu trả lời đúng:",
                        "audio_text": f"Hello, how are you today? I'm going to the shopping mall. Would you like to join me? We can have lunch together after shopping.",
                        "options": [
                            "The speaker is inviting someone to go shopping",
                            "The speaker is asking for directions to the mall",
                            "The speaker is talking about what they bought",
                            "The speaker is discussing lunch plans only"
                        ],
                        "correct_answer": "The speaker is inviting someone to go shopping",
                        "difficulty": difficulty,
                        "skill": "listening",
                        "points": 1
                    })
                elif i % 5 == 1:
                    # Câu hỏi viết
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "writing",
                        "question": f"Viết một đoạn văn ngắn (3-5 câu) về chủ đề: {random.choice(self.topics)}",
                        "correct_answer": "Đánh giá dựa trên: 1) Nội dung phù hợp với chủ đề, 2) Ngữ pháp chính xác, 3) Từ vựng phù hợp, 4) Cấu trúc câu đa dạng",
                        "difficulty": difficulty,
                        "skill": "writing",
                        "points": 2
                    })
                elif i % 5 == 2:
                    # Câu hỏi nói
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "speaking",
                        "question": f"Hãy nói về {random.choice(self.topics)} trong 30 giây.",
                        "correct_answer": "Đánh giá dựa trên: 1) Phát âm rõ ràng, 2) Ngữ điệu tự nhiên, 3) Sử dụng từ vựng phù hợp, 4) Nội dung liên quan đến chủ đề",
                        "difficulty": difficulty,
                        "skill": "speaking",
                        "points": 2
                    })
                elif i % 5 == 3:
                    # Câu hỏi đọc hiểu
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "reading",
                        "text": "Reading is an important skill for learning English. It helps you improve your vocabulary, grammar, and understanding of the language. Try to read something in English every day, even if it's just for a few minutes.",
                        "question": "Đâu là ý chính của đoạn văn trên?",
                        "options": [
                            "Đọc là một kỹ năng quan trọng khi học tiếng Anh",
                            "Học tiếng Anh rất khó",
                            "Nên học tiếng Anh mỗi ngày",
                            "Từ vựng quan trọng hơn ngữ pháp"
                        ],
                        "correct_answer": "Đọc là một kỹ năng quan trọng khi học tiếng Anh",
                        "difficulty": difficulty,
                        "skill": "reading",
                        "points": 1
                    })
                else:
                    # Câu hỏi trắc nghiệm ngữ pháp
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "grammar",
                        "question": "Chọn đáp án đúng để hoàn thành câu: She ___ to the store yesterday.",
                        "options": ["go", "goes", "went", "going"],
                        "correct_answer": "went",
                        "difficulty": difficulty,
                        "skill": "grammar",
                        "points": 1
                    })
        else:
            # Bài kiểm tra theo cấp độ cụ thể
            questions = []
            for i in range(length):
                # Chọn chủ đề phù hợp với cấp độ nếu không có chủ đề chỉ định
                if not topic and level in self.level_topics:
                    topic_options = self.level_topics[level]
                    current_topic = random.choice(topic_options)
                else:
                    current_topic = topic or random.choice(self.topics)
                
                if i % 4 == 0:
                    # Câu hỏi ngữ pháp
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "grammar",
                        "question": f"Chọn đáp án đúng để hoàn thành câu: She ___ to the store yesterday.",
                        "options": ["go", "goes", "went", "going"],
                        "correct_answer": "went",
                        "difficulty": level,
                        "skill": "grammar",
                        "points": 1
                    })
                elif i % 4 == 1:
                    # Câu hỏi từ vựng
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "vocabulary",
                        "question": f"Chọn từ đồng nghĩa với từ 'happy':",
                        "options": ["sad", "joyful", "angry", "tired"],
                        "correct_answer": "joyful",
                        "difficulty": level,
                        "skill": "vocabulary",
                        "points": 1
                    })
                elif i % 4 == 2:
                    # Câu hỏi nghe hiểu
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "listening",
                        "question": "Nghe đoạn hội thoại và trả lời câu hỏi: What is the woman going to do this weekend?",
                        "audio_text": "Man: Do you have any plans for the weekend? Woman: Yes, I'm going to visit my grandparents. They live in the countryside.",
                        "options": [
                            "Stay at home",
                            "Go shopping",
                            "Visit her grandparents",
                            "Work overtime"
                        ],
                        "correct_answer": "Visit her grandparents",
                        "difficulty": level,
                        "skill": "listening",
                        "points": 1
                    })
                else:
                    # Câu hỏi đọc hiểu theo chủ đề
                    questions.append({
                        "id": f"q{i+1}",
                        "type": "reading",
                        "text": f"The topic of {current_topic} is very interesting. Many people around the world enjoy learning about it. There are many books and websites dedicated to this subject.",
                        "question": f"What do many people enjoy?",
                        "options": [
                            f"Learning about {current_topic}",
                            "Writing books",
                            "Creating websites",
                            "Teaching others"
                        ],
                        "correct_answer": f"Learning about {current_topic}",
                        "difficulty": level,
                        "skill": "reading",
                        "points": 1
                    })
                    
        return {
            "questions": questions,
            "time_limit": 45 * length,  # 45 giây mỗi câu hỏi
            "passing_score": 7.0,
            "instructions": "Hãy đọc kỹ từng câu hỏi và chọn đáp án đúng. Đối với các câu hỏi nghe, hãy nghe kỹ file audio trước khi trả lời."
        }
    
    def generate_lesson(self, level, topic=None, weakness=None):
        """Tạo bài học dựa trên cấp độ, chủ đề và điểm yếu"""
        if not topic:
            # Chọn chủ đề phù hợp với cấp độ
            if level in self.level_topics:
                topic = random.choice(self.level_topics[level])
            else:
                topic = random.choice(self.topics)
            
        if self.model_loaded:
            # Xác định yêu cầu đặc biệt dựa trên điểm yếu
            special_requirements = ""
            if weakness:
                if "nghe" in weakness.lower() or "listening" in weakness.lower():
                    special_requirements = "Tập trung nhiều vào bài tập nghe hiểu và phân biệt âm."
                elif "nói" in weakness.lower() or "speaking" in weakness.lower():
                    special_requirements = "Tập trung nhiều vào bài tập luyện phát âm và giao tiếp."
                elif "đọc" in weakness.lower() or "reading" in weakness.lower():
                    special_requirements = "Tập trung nhiều vào bài tập đọc hiểu và mở rộng từ vựng."
                elif "viết" in weakness.lower() or "writing" in weakness.lower():
                    special_requirements = "Tập trung nhiều vào bài tập viết và cấu trúc câu."
                else:
                    special_requirements = f"Tập trung cải thiện điểm yếu: {weakness}"
            
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy tạo một bài học tiếng Anh hoàn chỉnh với các thông số sau:

Cấp độ: {level}
Chủ đề: {topic}
{"Điểm yếu cần cải thiện: " + weakness if weakness else ""}
{special_requirements}

Bài học cần bao gồm đầy đủ các phần: từ vựng, ngữ pháp, nghe, nói, đọc, viết. Bài học phải có các hoạt động thực hành cho cả 4 kỹ năng.

Vui lòng tạo bài học ở định dạng JSON theo cấu trúc sau:
```json
{{
  "title": "Tiêu đề bài học",
  "description": "Mô tả ngắn về bài học",
  "objectives": ["Mục tiêu 1", "Mục tiêu 2", ...],
  "sections": [
    {{
      "title": "Từ vựng",
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
      "title": "Luyện nghe",
      "content": "Hướng dẫn bài nghe",
      "listening_activities": [
        {{
          "title": "Hoạt động nghe",
          "audio_text": "Văn bản để tạo file audio",
          "questions": [
            {{
              "question": "Câu hỏi về bài nghe",
              "options": ["Lựa chọn A", "Lựa chọn B", ...],
              "correct_answer": "Đáp án đúng"
            }}
          ]
        }}
      ]
    }},
    {{
      "title": "Luyện nói",
      "content": "Hướng dẫn luyện nói",
      "speaking_activities": [
        {{
          "title": "Hoạt động nói",
          "instructions": "Hướng dẫn thực hiện",
          "phrases": ["Mẫu câu hữu ích 1", "Mẫu câu hữu ích 2", ...]
        }}
      ]
    }},
    {{
      "title": "Luyện đọc",
      "content": "Hướng dẫn đọc",
      "reading_text": "Văn bản đọc",
      "reading_questions": [
        {{
          "question": "Câu hỏi về bài đọc",
          "options": ["Lựa chọn A", "Lựa chọn B", ...],
          "correct_answer": "Đáp án đúng"
        }}
      ]
    }},
    {{
      "title": "Luyện viết",
      "content": "Hướng dẫn viết",
      "writing_activities": [
        {{
          "title": "Hoạt động viết",
          "instructions": "Hướng dẫn thực hiện",
          "example": "Ví dụ mẫu",
          "evaluation_criteria": ["Tiêu chí 1", "Tiêu chí 2", ...]
        }}
      ]
    }},
    {{
      "title": "Bài tập tổng hợp",
      "exercises": [
        {{
          "id": "ex1",
          "question": "Câu hỏi",
          "type": "loại bài tập",
          "options": ["Lựa chọn A", "Lựa chọn B", ...],
          "correct_answer": "Đáp án đúng"
        }}
      ]
    }}
  ]
}}
```

Lưu ý:
- Tạo nội dung phù hợp với cấp độ {level} và chủ đề {topic}
- Đảm bảo bài học có đủ 4 kỹ năng: nghe, nói, đọc, viết
- Với phần nghe, cung cấp "audio_text" để hệ thống tạo file audio
- Tạo các hoạt động thực hành tương tác thực tế và hữu ích
- Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo bài học
            response = self._generate_response(prompt, max_length=6000)
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
                "Phát triển kỹ năng nghe và nói trong tình huống thực tế",
                "Luyện đọc hiểu và viết liên quan đến chủ đề"
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
        
        # Phần luyện nghe
        listening_section = {
            "title": "Luyện nghe",
            "content": "<p>Hãy luyện tập kỹ năng nghe với bài tập sau:</p>",
            "listening_activities": [
                {
                    "title": "Nghe và trả lời câu hỏi",
                    "audio_text": f"This is a sample listening exercise about {topic}. Listen carefully and answer the questions.",
                    "questions": [
                        {
                            "question": "What is the main topic of the audio?",
                            "options": [f"{topic}", "Weather", "Food", "Travel"],
                            "correct_answer": f"{topic}"
                        }
                    ]
                }
            ]
        }
        
        # Phần luyện nói
        speaking_section = {
            "title": "Luyện nói",
            "content": "<p>Hãy luyện tập kỹ năng nói với hoạt động sau:</p>",
            "speaking_activities": [
                {
                    "title": "Thảo luận về chủ đề",
                    "instructions": f"Hãy nói về chủ đề {topic} trong 1-2 phút. Bạn có thể sử dụng các mẫu câu dưới đây.",
                    "phrases": [
                        f"I think {topic} is very interesting because...",
                        f"One important aspect of {topic} is...",
                        f"In my opinion, {topic} affects our daily lives by..."
                    ]
                }
            ]
        }
        
        # Phần luyện đọc
        reading_section = {
            "title": "Luyện đọc",
            "content": "<p>Hãy đọc đoạn văn sau và trả lời câu hỏi:</p>",
            "reading_text": f"This is a sample reading text about {topic}. It provides information about various aspects of {topic} and why it is important in our lives.",
            "reading_questions": [
                {
                    "question": "What is the main idea of the text?",
                    "options": [
                        f"The importance of {topic}",
                        "The history of education",
                        "The problems in society",
                        "The future technologies"
                    ],
                    "correct_answer": f"The importance of {topic}"
                }
            ]
        }
        
        # Phần luyện viết
        writing_section = {
            "title": "Luyện viết",
            "content": "<p>Hãy luyện tập kỹ năng viết với bài tập sau:</p>",
            "writing_activities": [
                {
                    "title": "Viết đoạn văn",
                    "instructions": f"Viết một đoạn văn ngắn (100-150 từ) về chủ đề {topic}. Sử dụng từ vựng và ngữ pháp đã học.",
                    "example": f"Here is a sample paragraph about {topic}: [Example paragraph]",
                    "evaluation_criteria": [
                        "Sử dụng từ vựng phù hợp",
                        "Cấu trúc câu đúng ngữ pháp",
                        "Nội dung phù hợp với chủ đề",
                        "Bố cục rõ ràng, mạch lạc"
                    ]
                }
            ]
        }
        
        # Phần bài tập tổng hợp
        practice_section = {
            "title": "Bài tập tổng hợp",
            "exercises": []
        }
        
        # Tạo các bài tập giả định
        for i in range(5):
            practice_section["exercises"].append({
                "id": f"ex{i+1}",
                "question": f"Câu hỏi luyện tập {i+1}",
                "type": random.choice(["multiple_choice", "fill_in_blank", "matching"]),
                "options": ["Option A", "Option B", "Option C", "Option D"] if i % 2 == 0 else None,
                "correct_answer": "Option A" if i % 2 == 0 else "Đáp án mẫu"
            })
        
        # Thêm các phần vào bài học
        lesson["sections"].extend([
            vocabulary_section, 
            grammar_section, 
            listening_section,
            speaking_section,
            reading_section,
            writing_section,
            practice_section
        ])
        
        return lesson
    
    def generate_feedback(self, answers, correct_answers):
        """Tạo phản hồi chi tiết cho bài tập"""
        if self.model_loaded:
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một giáo viên tiếng Anh chuyên nghiệp. Hãy đánh giá bài tập của học viên với các thông tin sau:

Câu trả lời của học viên: {json.dumps(answers, ensure_ascii=False)}
Câu trả lời đúng: {json.dumps(correct_answers, ensure_ascii=False)}

Vui lòng đánh giá và đưa ra phản hồi chi tiết ở định dạng JSON theo cấu trúc sau:
```json
{{
  "score": "Điểm số từ 0-10",
  "correct_count": "Số câu đúng",
  "total_questions": "Tổng số câu",
  "performance": "Đánh giá tổng quát (Xuất sắc/Tốt/Khá/Trung bình/Cần cải thiện)",
  "analysis": [
    {{
      "question_id": "Số thứ tự câu hỏi",
      "student_answer": "Câu trả lời của học viên",
      "correct_answer": "Câu trả lời đúng",
      "is_correct": true/false,
      "explanation": "Giải thích chi tiết"
    }}
  ],
  "strengths": ["Điểm mạnh 1", "Điểm mạnh 2", ...],
  "areas_to_improve": ["Cần cải thiện 1", "Cần cải thiện 2", ...],
  "suggestions": ["Gợi ý cải thiện 1", "Gợi ý cải thiện 2", ...],
  "next_steps": ["Bước tiếp theo 1", "Bước tiếp theo 2", ...]
}}
```

Hãy phân tích từng câu trả lời, chỉ ra lỗi cụ thể, nguyên nhân và cách sửa. Đưa ra nhận xét và gợi ý cụ thể, giúp học viên hiểu được điểm mạnh và điểm cần cải thiện. Phân tích cả các xu hướng lỗi để có gợi ý phù hợp.
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
        # Đếm số câu đúng đơn giản
        correct_count = sum([1 for i, ans in enumerate(answers) if ans == correct_answers[i]]) if len(answers) == len(correct_answers) else 0
        score = (correct_count / len(answers)) * 10 if answers else 0
        
        # Đánh giá hiệu suất
        if score >= 9:
            performance = "Xuất sắc"
        elif score >= 8:
            performance = "Tốt"
        elif score >= 7:
            performance = "Khá"
        elif score >= 5:
            performance = "Trung bình"
        else:
            performance = "Cần cải thiện"
        
        # Tạo phản hồi
        feedback = {
            "score": score,
            "correct_count": correct_count,
            "total_questions": len(answers),
            "performance": performance,
            "analysis": [
                {
                    "question_id": i+1,
                    "student_answer": ans,
                    "correct_answer": correct_answers[i] if i < len(correct_answers) else "Không có đáp án",
                    "is_correct": ans == correct_answers[i] if i < len(correct_answers) else False,
                    "explanation": "Bạn đã trả lời đúng!" if (i < len(correct_answers) and ans == correct_answers[i]) else "Đáp án của bạn chưa đúng."
                } for i, ans in enumerate(answers)
            ],
            "strengths": [
                "Bạn làm tốt các câu hỏi cơ bản" if score >= 5 else "Bạn đã có nỗ lực trong bài làm",
                "Bạn có kiến thức tốt về chủ đề này" if score >= 7 else "Bạn có thể cải thiện thêm"
            ],
            "areas_to_improve": [
                "Cần ôn lại kiến thức cơ bản" if score < 7 else "Cần luyện tập thêm các dạng câu hỏi nâng cao",
                "Cần đọc kỹ hơn các câu hỏi trước khi trả lời" if score < 8 else "Cần mở rộng vốn từ vựng"
            ],
            "suggestions": [
                "Xem lại lý thuyết và ví dụ trong bài học" if score < 7 else "Thử thách bản thân với các bài tập nâng cao",
                "Luyện tập thêm với các bài tập tương tự" if score < 8 else "Tìm hiểu thêm các tài liệu nâng cao"
            ],
            "next_steps": [
                "Làm lại bài tập này sau khi ôn tập" if score < 7 else "Chuyển sang bài học tiếp theo",
                "Tham khảo thêm các nguồn học liệu bổ sung" if score < 8 else "Thử thách bản thân với các bài tập khó hơn"
            ]
        }
        
        return feedback
    
    def generate_course(self, level, weaknesses=None, user_profile=None):
        """Tạo khóa học cá nhân hóa dựa trên cấp độ, điểm yếu và hồ sơ người dùng"""
        if self.model_loaded:
            # Tạo phần mô tả điểm yếu
            weakness_desc = ""
            if weaknesses:
                weakness_desc = f"Điểm yếu cần cải thiện: {', '.join(weaknesses)}"
            
            # Tạo phần mô tả hồ sơ người dùng
            profile_desc = ""
            if user_profile:
                if 'learning_goals' in user_profile:
                    profile_desc += f"Mục tiêu học tập: {user_profile['learning_goals']}\n"
                if 'preferred_topics' in user_profile:
                    profile_desc += f"Chủ đề yêu thích: {user_profile['preferred_topics']}\n"
            
            # Tạo prompt cho mô hình
            prompt = f"""Bạn là một chuyên gia giáo dục sáng tạo. Hãy thiết kế một khóa học tiếng Anh cá nhân hóa theo các thông số sau:

Cấp độ: {level}
{weakness_desc}
{profile_desc}

Vui lòng tạo khóa học ở định dạng JSON theo cấu trúc sau:
```json
{{
  "title": "Tiêu đề khóa học",
  "description": "Mô tả chi tiết về khóa học",
  "level": "{level}",
  "duration_weeks": 8,
  "topic": "Chủ đề chính của khóa học",
  "lessons": [
    {{
      "title": "Tiêu đề bài học 1",
      "order": 1,
      "description": "Mô tả ngắn về bài học",
      "topic": "Chủ đề bài học",
      "estimated_time": 45,
      "focus_skills": ["Kỹ năng trọng tâm 1", "Kỹ năng trọng tâm 2"],
      "objectives": ["Mục tiêu 1", "Mục tiêu 2"]
    }},
    ...
  ],
  "assessments": [
    {{
      "title": "Tiêu đề bài kiểm tra",
      "description": "Mô tả bài kiểm tra",
      "week": 4,
      "type": "midterm/final/quiz",
      "focus_areas": ["Lĩnh vực kiểm tra 1", "Lĩnh vực kiểm tra 2"]
    }}
  ],
  "recommendations": [
    {{
      "type": "resource/activity",
      "title": "Tiêu đề tài nguyên/hoạt động",
      "description": "Mô tả tài nguyên/hoạt động",
      "url": "Đường dẫn (nếu có)"
    }}
  ]
}}
```

Lưu ý:
- Khóa học phải cá nhân hóa dựa trên điểm yếu và hồ sơ người dùng (nếu có)
- Thời lượng khóa học nên là 8 tuần với 2-3 bài học mỗi tuần
- Các bài học phải bao gồm đầy đủ 4 kỹ năng: nghe, nói, đọc, viết
- Mỗi bài học tập trung vào một chủ đề cụ thể và có mức độ khó tăng dần
- Phải có các bài kiểm tra định kỳ (ít nhất 1 bài giữa kỳ và 1 bài cuối kỳ)
- Đề xuất các tài nguyên và hoạt động bổ sung phù hợp
- Tạo ít nhất 12 bài học cho khóa học
- Khóa học phải phù hợp với cấp độ {level} và hướng đến cấp độ tiếp theo
- Chỉ trả về JSON, không cần giải thích thêm."""

            # Gọi mô hình để tạo khóa học
            response = self._generate_response(prompt, max_length=8000)
            course_data = self._extract_json_from_text(response)
            
            if course_data:
                # Thêm trường is_auto_generated
                course_data["is_auto_generated"] = True
                course_data["created_at"] = datetime.utcnow().isoformat()
                return course_data
        
        # Nếu không thể sử dụng mô hình hoặc kết quả không hợp lệ, sử dụng phương pháp dự phòng
        # Xác định topics phù hợp với level
        if level in ["Beginner (A1)", "Elementary (A2)"]:
            topics = ["Greetings", "Family", "Food", "Daily Activities", "Numbers", "Colors", "Weather"]
            title = f"Khóa học tiếng Anh cơ bản cho người mới bắt đầu ({level})"
            description = f"Khóa học này được tạo tự động dựa trên kết quả bài test. Thiết kế đặc biệt cho trình độ {level}, giúp bạn nắm vững các kiến thức cơ bản về tiếng Anh."
        elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
            topics = ["Travel", "Work", "Culture", "Media", "Environment", "Technology", "Education"]
            title = f"Tiếng Anh giao tiếp trung cấp ({level})"
            description = f"Khóa học được AI tạo riêng dựa vào kết quả bài kiểm tra. Tập trung nâng cao khả năng giao tiếp tiếng Anh cho công việc và đời sống hàng ngày ở trình độ {level}."
        else:  # C1, C2
            topics = ["Business", "Academic", "Literature", "Global Issues", "Science", "Advanced Media", "Professional Communication"]
            title = f"Tiếng Anh nâng cao - chuyên sâu ({level})"
            description = f"Khóa học chuyên sâu được cá nhân hóa theo trình độ {level}, giúp bạn làm chủ tiếng Anh ở cấp độ gần với người bản xứ."
        
        # Xác định tập trung vào kỹ năng cần cải thiện nếu có
        focus_skills = ["listening", "speaking", "reading", "writing"]
        if weaknesses:
            # Sắp xếp focus_skills để đặt các kỹ năng yếu lên đầu
            for skill in focus_skills.copy():
                if any(skill in weakness.lower() for weakness in weaknesses):
                    focus_skills.remove(skill)
                    focus_skills.insert(0, skill)
        
        # Tạo các bài học
        lessons = []
        for i, topic in enumerate(topics[:12], 1):
            # Xác định kỹ năng trọng tâm cho bài học này
            skill_index = i % len(focus_skills)
            primary_skill = focus_skills[skill_index]
            secondary_skill = focus_skills[(skill_index + 1) % len(focus_skills)]
            
            lesson = {
                "title": f"Bài {i}: {topic}",
                "order": i,
                "description": f"Bài học về chủ đề {topic} tập trung vào kỹ năng {primary_skill} và {secondary_skill}",
                "topic": topic,
                "estimated_time": 45,
                "focus_skills": [primary_skill, secondary_skill],
                "objectives": [
                    f"Học từ vựng liên quan đến {topic}",
                    f"Luyện tập {primary_skill} trong tình huống thực tế",
                    f"Phát triển kỹ năng {secondary_skill}"
                ]
            }
            lessons.append(lesson)
        
        # Tạo các bài kiểm tra
        assessments = [
            {
                "title": "Kiểm tra giữa kỳ",
                "description": "Đánh giá tiến độ học tập sau 4 tuần đầu tiên",
                "week": 4,
                "type": "midterm",
                "focus_areas": topics[:4]
            },
            {
                "title": "Kiểm tra cuối kỳ",
                "description": "Đánh giá toàn diện các kiến thức và kỹ năng đã học",
                "week": 8,
                "type": "final",
                "focus_areas": topics
            }
        ]
        
        # Tạo các tài nguyên bổ sung
        recommendations = [
            {
                "type": "resource",
                "title": "Tài liệu học tập bổ sung",
                "description": "Sách và tài liệu để nâng cao vốn từ vựng và ngữ pháp",
                "url": "#"
            },
            {
                "type": "activity",
                "title": "Câu lạc bộ tiếng Anh",
                "description": "Tham gia các hoạt động giao tiếp tiếng Anh trực tuyến",
                "url": "#"
            }
        ]
        
        # Cấu trúc khóa học
        course = {
            "title": title,
            "description": description,
            "level": level,
            "duration_weeks": 8,
            "topic": random.choice(topics),
            "is_auto_generated": True,
            "lessons": lessons,
            "assessments": assessments,
            "recommendations": recommendations,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return course

# Hệ thống cache nội dung
class ContentCache:
    """
    Hệ thống quản lý cache để lưu trữ và tự động cập nhật nội dung đã tạo sẵn
    bao gồm bài test, bài học và khóa học để tránh phải tạo mới mỗi lần có yêu cầu
    """
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.lock = threading.Lock()
        self.cache_dir = 'content_cache'
        self.cache_version = 1  # Tăng phiên bản khi thay đổi cấu trúc cache
        
        # Đảm bảo thư mục cache tồn tại
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Khởi tạo các cache cho từng loại nội dung
        self.test_cache = {
            'placement': [],  # Lưu các bài test xếp loại
            'level': {  # Lưu các bài test theo cấp độ
                "Beginner (A1)": [],
                "Elementary (A2)": [],
                "Intermediate (B1)": [],
                "Upper Intermediate (B2)": [],
                "Advanced (C1)": [],
                "Proficient (C2)": []
            },
            'topics': {}  # Lưu các bài test theo chủ đề và cấp độ
        }
        
        self.lesson_cache = {
            "Beginner (A1)": {},
            "Elementary (A2)": {},
            "Intermediate (B1)": {},
            "Upper Intermediate (B2)": {},
            "Advanced (C1)": {},
            "Proficient (C2)": {}
        }
        
        self.course_cache = {
            "Beginner (A1)": [],
            "Elementary (A2)": [],
            "Intermediate (B1)": [],
            "Upper Intermediate (B2)": [],
            "Advanced (C1)": [],
            "Proficient (C2)": []
        }
        
        # Cache cho media
        self.media_cache = {
            'audio': {},  # map: text -> đường dẫn file audio
            'images': {}  # map: id -> đường dẫn file hình ảnh
        }
        
        # Tải cache từ đĩa nếu có
        self.load_cache()
        
        # Tạo thread chạy scheduler
        self.scheduler_thread = None
        self.scheduler_running = False
    
    def initialize_remaining_cache(self):
        """Khởi tạo phần còn lại của cache sau khi đã tạo dữ liệu cần thiết"""
        thread = threading.Thread(target=self._initialize_remaining_cache)
        thread.daemon = True
        thread.start()
        return thread

    def _initialize_remaining_cache(self):
        """Công việc khởi tạo phần còn lại của cache"""
        print("Bắt đầu khởi tạo cache nội dung bổ sung...")
        
        # Tạo thêm bài test xếp loại nếu chưa đủ
        if len(self.test_cache['placement']) < 3:
            self._generate_placement_tests(3 - len(self.test_cache['placement']))
        
        # Tạo các bài test theo cấp độ
        for level in self.ai.levels:
            if level not in self.test_cache['level'] or len(self.test_cache['level'][level]) < 2:
                count = 2 - len(self.test_cache['level'].get(level, []))
                if count > 0:
                    self._generate_level_tests(level, count)
        
        # Tạo các bài học mẫu cho mỗi cấp độ và chủ đề
        for level in self.ai.levels:
            # Chỉ tạo cho 3 chủ đề mỗi cấp độ để tiết kiệm thời gian
            for topic in random.sample(self.ai.topics, min(3, len(self.ai.topics))):
                if level not in self.lesson_cache or topic not in self.lesson_cache.get(level, {}):
                    self._generate_lesson(level, topic)
        
        # Tạo mẫu khóa học cho mỗi cấp độ
        for level in self.ai.levels:
            if level not in self.course_cache or not self.course_cache[level]:
                self._generate_course_templates(level, 1)
        
        # Tạo sẵn một số media mẫu
        self._generate_sample_media()
        
        # Lưu cache
        self.save_cache()
        print("Đã hoàn thành khởi tạo cache nội dung bổ sung!")    
    
    def _generate_sample_media(self):
        """Tạo sẵn các media mẫu để sử dụng trong bài học và bài test"""
        print("Đang tạo các media mẫu...")
        
        # Tạo thư mục lưu trữ nếu chưa có
        audio_dir = os.path.join('static', 'audio', 'samples')
        images_dir = os.path.join('static', 'images', 'samples')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Tạo câu tiếng Anh mới sử dụng DeepSeek AI
        print("Đang tạo các câu nói và bài đọc mới bằng AI...")
        
        try:
            # Tạo các câu ngắn mới cho bài nghe bằng AI thay vì sử dụng danh sách cố định
            common_phrases = []
            topics = ["greeting", "introduction", "daily life", "hobbies", "shopping", "travel", "food", "directions", "weather", "health"]
            
            for topic in topics[:5]:  # Giới hạn ở 5 chủ đề để tiết kiệm thời gian
                prompt = f"Tạo một câu tiếng Anh ngắn và thông dụng về chủ đề '{topic}'. Chỉ trả về câu tiếng Anh, không kèm giải thích."
                response = self.ai._generate_response(prompt, max_length=100, temperature=0.7)
                
                # Làm sạch câu trả về từ AI
                if response:
                    # Loại bỏ dấu ngoặc kép và dấu kết thúc câu nếu có
                    sentence = response.strip('"\'.,\n')
                    # Đảm bảo câu có dấu kết thúc
                    if not sentence.endswith(('.', '?', '!')):
                        sentence += '.'
                    common_phrases.append(sentence)
                    print(f"Đã tạo câu mới: {sentence}")
            
            # Nếu không có câu nào được tạo, sử dụng một số câu mặc định
            if not common_phrases:
                common_phrases = [
                    "Hello, how are you today?",
                    "My name is Anna and I'm a college student.",
                    "Where do you live in this beautiful city?",
                    "I enjoy reading books and watching documentaries."
                ]
                print("Sử dụng các câu mặc định do không tạo được câu mới bằng AI")
            
            # Tạo các đoạn văn bài đọc mới bằng AI
            reading_texts = []
            reading_topics = ["language learning", "education", "technology", "culture", "environment"]
            
            for topic in reading_topics[:2]:  # Giới hạn ở 2 chủ đề
                prompt = f"Tạo một đoạn văn ngắn (3-4 câu) về chủ đề '{topic}' để làm bài tập đọc tiếng Anh cho học sinh. Chỉ trả về đoạn văn, không kèm giải thích."
                response = self.ai._generate_response(prompt, max_length=500, temperature=0.7)
                
                if response:
                    # Làm sạch đoạn văn
                    paragraph = response.strip('"\'').strip()
                    reading_texts.append(paragraph)
                    print(f"Đã tạo đoạn văn mới về chủ đề: {topic}")
            
            # Nếu không có đoạn văn nào được tạo, sử dụng đoạn văn mặc định
            if not reading_texts:
                reading_texts = [
                    "Language learning is a journey that requires patience and dedication. Every new word and grammar rule builds your understanding. With consistent practice, you'll find yourself making progress each day."
                ]
                print("Sử dụng đoạn văn mặc định do không tạo được đoạn văn mới bằng AI")
                
        except Exception as e:
            print(f"Lỗi khi tạo nội dung mới bằng AI: {str(e)}")
            # Sử dụng một số câu và đoạn văn mặc định trong trường hợp có lỗi
            common_phrases = [
                "Hello, how are you?",
                "My name is John. I am a student.",
                "Where do you live?",
                "I like playing tennis and swimming."
            ]
            
            reading_texts = [
                "Learning a new language can be challenging but also very rewarding. When you learn a new language, you gain new perspectives and opportunities."
            ]
            print("Sử dụng nội dung mặc định do lỗi khi tạo nội dung mới")
        
        # Tạo các file audio mẫu
        for i, phrase in enumerate(common_phrases):
            audio_path = os.path.join(audio_dir, f'phrase_{i+1}.mp3')
            
            # Chỉ tạo nếu file chưa tồn tại
            if not os.path.exists(audio_path):
                TienIchMedia.tao_audio_tu_van_ban(phrase, audio_path)
            
            # Lưu vào cache
            self.media_cache['audio'][phrase] = audio_path
        
        # Tạo các hình ảnh bài đọc mẫu
        for i, text in enumerate(reading_texts):
            image_path = os.path.join(images_dir, f'reading_{i+1}.png')
            
            # Chỉ tạo nếu file chưa tồn tại
            if not os.path.exists(image_path):
                TienIchMedia.tao_hinh_anh_van_ban(text, image_path)
            
            # Lưu vào cache
            self.media_cache['images'][f'reading_{i+1}'] = image_path
        
        # Tạo một số hình ảnh biểu đồ tiến bộ mẫu
        for i in range(2):
            chart_path = os.path.join(images_dir, f'progress_chart_{i+1}.png')
            
            # Chỉ tạo nếu file chưa tồn tại
            if not os.path.exists(chart_path):
                # Tạo dữ liệu giả định
                chart_data = {
                    'labels': [f'Ngày {j+1}' for j in range(7)],
                    'values': [random.randint(60, 95) for _ in range(7)]
                }
                TienIchMedia.tao_bieu_do_tien_bo(chart_data, chart_path)
            
            # Lưu vào cache
            self.media_cache['images'][f'progress_chart_{i+1}'] = chart_path
    
    def load_cache(self):
        """Tải cache từ đĩa"""
        try:
            # Tải cache bài test
            test_cache_path = os.path.join(self.cache_dir, 'test_cache.json')
            if os.path.exists(test_cache_path):
                with open(test_cache_path, 'r', encoding='utf-8') as f:
                    self.test_cache = json.load(f)
                print(f"Đã tải cache bài test từ {test_cache_path}")
            
            # Tải cache bài học
            lesson_cache_path = os.path.join(self.cache_dir, 'lesson_cache.json')
            if os.path.exists(lesson_cache_path):
                with open(lesson_cache_path, 'r', encoding='utf-8') as f:
                    self.lesson_cache = json.load(f)
                print(f"Đã tải cache bài học từ {lesson_cache_path}")
            
            # Tải cache khóa học
            course_cache_path = os.path.join(self.cache_dir, 'course_cache.json')
            if os.path.exists(course_cache_path):
                with open(course_cache_path, 'r', encoding='utf-8') as f:
                    self.course_cache = json.load(f)
                print(f"Đã tải cache khóa học từ {course_cache_path}")
            
            # Tải cache media
            media_cache_path = os.path.join(self.cache_dir, 'media_cache.json')
            if os.path.exists(media_cache_path):
                with open(media_cache_path, 'r', encoding='utf-8') as f:
                    self.media_cache = json.load(f)
                print(f"Đã tải cache media từ {media_cache_path}")
            
            # Kiểm tra số lượng bài test đã cache
            placement_count = len(self.test_cache['placement'])
            level_counts = {level: len(tests) for level, tests in self.test_cache['level'].items()}
            
            print(f"Số lượng bài test xếp loại: {placement_count}")
            print(f"Số lượng bài test theo cấp độ: {level_counts}")
            
        except Exception as e:
            print(f"Lỗi khi tải cache: {str(e)}")
            # Khởi tạo cache mới nếu có lỗi
            print("Khởi tạo cache mới...")
    
    def save_cache(self):
        """Lưu cache vào đĩa"""
        try:
            # Lưu cache bài test
            test_cache_path = os.path.join(self.cache_dir, 'test_cache.json')
            with open(test_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_cache, f, ensure_ascii=False, indent=2)
            
            # Lưu cache bài học
            lesson_cache_path = os.path.join(self.cache_dir, 'lesson_cache.json')
            with open(lesson_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.lesson_cache, f, ensure_ascii=False, indent=2)
            
            # Lưu cache khóa học
            course_cache_path = os.path.join(self.cache_dir, 'course_cache.json')
            with open(course_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.course_cache, f, ensure_ascii=False, indent=2)
            
            # Lưu cache media
            media_cache_path = os.path.join(self.cache_dir, 'media_cache.json')
            with open(media_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.media_cache, f, ensure_ascii=False, indent=2)
            
            print(f"Đã lưu cache thành công vào {self.cache_dir}")
            
        except Exception as e:
            print(f"Lỗi khi lưu cache: {str(e)}")

    def get_placement_test(self):
        """Lấy một bài test xếp loại từ cache hoặc tạo mới nếu chưa có"""
        with self.lock:
            if not self.test_cache['placement']:
                # Tạo ngay lập tức một bài test để sử dụng
                print("Đang tạo bài test xếp loại mới (ưu tiên)...")
                test_data = self.ai.generate_test()
                
                # Xử lý media cho bài test
                test_data = self._process_test_media(test_data)
                
                self.test_cache['placement'].append(test_data)
                self.save_cache()
                return test_data
            
            # Trả về bài test từ cache và đảm bảo đã có media
            test_data = self.test_cache['placement'][0]
            return self._process_test_media(test_data)
    
    def get_level_test(self, level):
        """Lấy một bài test theo cấp độ từ cache"""
        with self.lock:
            if level not in self.test_cache['level'] or not self.test_cache['level'][level]:
                # Tạo mới nếu cache trống
                print(f"Cache bài test cấp độ {level} trống, đang tạo mới...")
                self._generate_level_tests(level, 1)
            
            # Trả về bài test theo cấp độ từ cache
            if self.test_cache['level'][level]:
                test_data = random.choice(self.test_cache['level'][level])
                return self._process_test_media(test_data)
            else:
                return None
    
    def get_topic_test(self, level, topic, length=10):
        """Lấy một bài test theo chủ đề và cấp độ từ cache"""
        with self.lock:
            # Tạo key cho topic
            topic_key = f"{level}_{topic}"
            
            if 'topics' not in self.test_cache:
                self.test_cache['topics'] = {}
                
            if topic_key not in self.test_cache['topics'] or not self.test_cache['topics'][topic_key]:
                # Tạo mới nếu cache trống
                print(f"Cache bài test chủ đề {topic} cấp độ {level} trống, đang tạo mới...")
                test_data = self.ai.generate_test(level, topic, length)
                
                # Xử lý media cho bài test
                test_data = self._process_test_media(test_data)
                
                if 'topics' not in self.test_cache:
                    self.test_cache['topics'] = {}
                
                if topic_key not in self.test_cache['topics']:
                    self.test_cache['topics'][topic_key] = []
                
                self.test_cache['topics'][topic_key].append(test_data)
                self.save_cache()
            
            # Trả về bài test theo chủ đề từ cache
            if self.test_cache['topics'].get(topic_key):
                test_data = random.choice(self.test_cache['topics'][topic_key])
                return self._process_test_media(test_data)
            else:
                return None
    
    def _process_test_media(self, test_data):
        """Xử lý media cho bài test (tạo audio, hình ảnh nếu cần)"""
        if not test_data or 'questions' not in test_data:
            return test_data
        
        # Thư mục lưu trữ media
        audio_dir = os.path.join('static', 'audio', 'tests')
        images_dir = os.path.join('static', 'images', 'tests')
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Duyệt qua các câu hỏi và tạo media nếu cần
        for question in test_data['questions']:
            # Xử lý câu hỏi nghe
            if question.get('type') == 'listening' and 'audio_text' in question:
                audio_text = question['audio_text']
                
                # Tạo ID duy nhất cho file audio dựa trên nội dung
                audio_id = f"audio_{hash(audio_text) % 10000:04d}"
                audio_path = os.path.join(audio_dir, f"{audio_id}.mp3")
                
                # Kiểm tra xem đã có trong cache chưa
                if audio_text in self.media_cache.get('audio', {}):
                    question['audio_url'] = self.media_cache['audio'][audio_text]
                # Nếu file chưa tồn tại, tạo mới
                elif not os.path.exists(audio_path):
                    TienIchMedia.tao_audio_tu_van_ban(audio_text, audio_path)
                    # Lưu vào cache
                    if 'audio' not in self.media_cache:
                        self.media_cache['audio'] = {}
                    self.media_cache['audio'][audio_text] = audio_path
                    question['audio_url'] = audio_path
                else:
                    question['audio_url'] = audio_path
            
            # Xử lý câu hỏi đọc nếu có text dài
            if question.get('type') == 'reading' and 'text' in question and len(question['text']) > 100:
                text = question['text']
                
                # Tạo ID duy nhất cho hình ảnh
                image_id = f"reading_{hash(text) % 10000:04d}"
                image_path = os.path.join(images_dir, f"{image_id}.png")
                
                # Nếu file chưa tồn tại, tạo mới
                if not os.path.exists(image_path):
                    TienIchMedia.tao_hinh_anh_van_ban(text, image_path, 
                                                 tieu_de="Reading Exercise")
                    # Lưu vào cache
                    if 'images' not in self.media_cache:
                        self.media_cache['images'] = {}
                    self.media_cache['images'][image_id] = image_path
                
                # Thêm đường dẫn hình ảnh vào câu hỏi
                question['image_url'] = image_path
        
        # Xác định phiên bản đã xử lý để tránh xử lý lại
        test_data['media_processed'] = True
        
        return test_data
    
    def get_lesson(self, level, topic=None, weakness=None):
        """Lấy một bài học theo cấp độ và chủ đề từ cache"""
        with self.lock:
            # Nếu không có chủ đề, chọn ngẫu nhiên một chủ đề phù hợp với cấp độ
            if not topic:
                if level in self.ai.level_topics:
                    topic = random.choice(self.ai.level_topics[level])
                else:
                    topic = random.choice(self.ai.topics)
            
            topic_key = topic
            if weakness:
                topic_key = f"{topic}_{weakness}"
                
            if (level not in self.lesson_cache or 
                topic_key not in self.lesson_cache[level]):
                # Tạo mới nếu cache trống
                print(f"Cache bài học {topic_key} cấp độ {level} trống, đang tạo mới...")
                lesson_data = self.ai.generate_lesson(level, topic, weakness)
                
                # Xử lý media cho bài học
                lesson_data = self._process_lesson_media(lesson_data, level, topic)
                
                if level not in self.lesson_cache:
                    self.lesson_cache[level] = {}
                
                self.lesson_cache[level][topic_key] = lesson_data
                self.save_cache()
            
            # Trả về bài học từ cache
            lesson_data = self.lesson_cache[level].get(topic_key)
            # Đảm bảo media đã được xử lý
            if lesson_data and not lesson_data.get('media_processed'):
                lesson_data = self._process_lesson_media(lesson_data, level, topic)
                self.lesson_cache[level][topic_key] = lesson_data
                self.save_cache()
            
            return lesson_data
    
    def _process_lesson_media(self, lesson_data, level, topic):
        """Xử lý media cho bài học (tạo audio, hình ảnh cho các phần nội dung)"""
        if not lesson_data:
            return lesson_data
        
        # Thư mục lưu trữ media cho bài học
        level_dir = level.split()[0].lower()  # Lấy phần đầu của cấp độ (beginner, elementary,...)
        topic_dir = topic.lower().replace(' ', '_')
        
        audio_dir = os.path.join('static', 'audio', 'lessons', level_dir, topic_dir)
        images_dir = os.path.join('static', 'images', 'lessons', level_dir, topic_dir)
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Xử lý các phần trong bài học
        if 'sections' in lesson_data:
            for i, section in enumerate(lesson_data['sections']):
                section_type = section.get('title', '').lower()
                
                # Xử lý phần từ vựng - tạo hình ảnh danh sách từ
                if 'vocabulary_list' in section and len(section['vocabulary_list']) > 0:
                    image_id = f"vocab_{i}_{hash(topic) % 1000:03d}"
                    image_path = os.path.join(images_dir, f"{image_id}.png")
                    
                    if not os.path.exists(image_path):
                        TienIchMedia.tao_hinh_anh_tu_vung(section['vocabulary_list'], image_path)
                    
                    # Thêm đường dẫn hình ảnh vào phần từ vựng
                    section['vocabulary_image'] = image_path
                
                # Xử lý phần ngữ pháp - tạo hình ảnh minh họa ngữ pháp
                if 'grammar_points' in section and len(section['grammar_points']) > 0:
                    image_id = f"grammar_{i}_{hash(topic) % 1000:03d}"
                    image_path = os.path.join(images_dir, f"{image_id}.png")
                    
                    if not os.path.exists(image_path):
                        TienIchMedia.tao_hinh_anh_ngu_phap(section['grammar_points'], image_path)
                    
                    # Thêm đường dẫn hình ảnh vào phần ngữ pháp
                    section['grammar_image'] = image_path
                
                # Xử lý phần nghe - tạo file audio
                if 'listening_activities' in section:
                    for j, activity in enumerate(section['listening_activities']):
                        if 'audio_text' in activity:
                            audio_text = activity['audio_text']
                            audio_id = f"listening_{i}_{j}_{hash(audio_text) % 1000:03d}"
                            audio_path = os.path.join(audio_dir, f"{audio_id}.mp3")
                            
                            if not os.path.exists(audio_path):
                                TienIchMedia.tao_audio_tu_van_ban(audio_text, audio_path)
                            
                            # Thêm đường dẫn audio vào hoạt động nghe
                            activity['audio_url'] = audio_path
                
                # Xử lý phần đọc - tạo hình ảnh cho văn bản đọc
                if 'reading_text' in section and len(section['reading_text']) > 100:
                    reading_text = section['reading_text']
                    image_id = f"reading_{i}_{hash(reading_text) % 1000:03d}"
                    image_path = os.path.join(images_dir, f"{image_id}.png")
                    
                    if not os.path.exists(image_path):
                        TienIchMedia.tao_hinh_anh_van_ban(reading_text, image_path, 
                                                     tieu_de="Reading Exercise")
                    
                    # Thêm đường dẫn hình ảnh vào phần đọc
                    section['reading_image'] = image_path
        
        # Đánh dấu đã xử lý media
        lesson_data['media_processed'] = True
        
        return lesson_data
    
    def get_course(self, level, user_id=None, weaknesses=None):
        """Lấy một khóa học theo cấp độ từ cache hoặc tạo mới"""
        # Ưu tiên trả về khóa học đã có trong cache
        with self.lock:
            if level in self.course_cache and self.course_cache[level]:
                # Chọn ngẫu nhiên một khóa học và sao chép để tránh xung đột
                course_data = random.choice(self.course_cache[level]).copy()
                # Tùy chỉnh một chút để tạo sự đa dạng
                course_data['creation_time'] = datetime.utcnow().isoformat()
                if user_id:
                    course_data['created_for'] = user_id
                return course_data
            else:
                # Nếu không có trong cache, tạo mới và lưu vào cache
                return self._generate_course_template(level, weaknesses, user_id)
    
    def initialize_cache(self, background=True):
        """Khởi tạo cache với các nội dung cơ bản"""
        # Kiểm tra nếu cache đã có dữ liệu
        has_data = False
        if (len(self.test_cache['placement']) > 0 or 
            any(len(tests) > 0 for tests in self.test_cache['level'].values())):
            has_data = True
            print("Cache đã có dữ liệu, bỏ qua khởi tạo...")
            return
        
        if background:
            # Chạy trong nền
            thread = threading.Thread(target=self._initialize_cache_task)
            thread.daemon = True
            thread.start()
            return thread
        else:
            # Chạy đồng bộ
            return self._initialize_cache_task()
    
    def _initialize_cache_task(self):
        """Công việc khởi tạo cache, có thể chạy trong nền"""
        print("Bắt đầu khởi tạo cache nội dung...")
        
        # Tạo các bài test xếp loại
        self._generate_placement_tests(3)
        
        # Tạo các bài test theo cấp độ
        for level in self.ai.levels:
            self._generate_level_tests(level, 2)
        
        # Tạo các bài học mẫu cho mỗi cấp độ và chủ đề
        for level in self.ai.levels:
            # Chỉ tạo cho 3 chủ đề mỗi cấp độ để tiết kiệm thời gian
            for topic in random.sample(self.ai.topics, 3):
                self._generate_lesson(level, topic)
        
        # Tạo mẫu khóa học cho mỗi cấp độ
        for level in self.ai.levels:
            self._generate_course_templates(level, 1)
        
        # Tạo sẵn một số file media mẫu
        self._generate_sample_media()
        
        # Lưu cache
        self.save_cache()
        print("Đã hoàn thành khởi tạo cache nội dung!")
    
    def _generate_placement_tests(self, count=1):
        """Tạo các bài test xếp loại và lưu vào cache"""
        print(f"Đang tạo {count} bài test xếp loại...")
        for _ in range(count):
            try:
                test_data = self.ai.generate_test()
                # Xử lý media cho bài test
                test_data = self._process_test_media(test_data)
                self.test_cache['placement'].append(test_data)
                print(f"Đã tạo thành công bài test xếp loại #{len(self.test_cache['placement'])}")
            except Exception as e:
                print(f"Lỗi khi tạo bài test xếp loại: {str(e)}")
        
        # Giới hạn số lượng bài test xếp loại
        self.test_cache['placement'] = self.test_cache['placement'][-10:]  # Giữ tối đa 10 bài
        self.save_cache()
    
    def _generate_level_tests(self, level, count=1):
        """Tạo các bài test theo cấp độ và lưu vào cache"""
        print(f"Đang tạo {count} bài test cấp độ {level}...")
        if level not in self.test_cache['level']:
            self.test_cache['level'][level] = []
            
        for _ in range(count):
            try:
                test_data = self.ai.generate_test(level)
                # Xử lý media cho bài test
                test_data = self._process_test_media(test_data)
                self.test_cache['level'][level].append(test_data)
                print(f"Đã tạo thành công bài test cấp độ {level} #{len(self.test_cache['level'][level])}")
            except Exception as e:
                print(f"Lỗi khi tạo bài test cấp độ {level}: {str(e)}")
        
        # Giới hạn số lượng bài test theo cấp độ
        self.test_cache['level'][level] = self.test_cache['level'][level][-5:]  # Giữ tối đa 5 bài mỗi cấp độ
        self.save_cache()
    
    def _generate_lesson(self, level, topic, weakness=None):
        """Tạo một bài học và lưu vào cache"""
        print(f"Đang tạo bài học chủ đề {topic} cấp độ {level}...")
        try:
            lesson_data = self.ai.generate_lesson(level, topic, weakness)
            
            # Xử lý media cho bài học
            lesson_data = self._process_lesson_media(lesson_data, level, topic)
            
            if level not in self.lesson_cache:
                self.lesson_cache[level] = {}
            
            topic_key = topic
            if weakness:
                topic_key = f"{topic}_{weakness}"
                
            self.lesson_cache[level][topic_key] = lesson_data
            print(f"Đã tạo thành công bài học chủ đề {topic_key} cấp độ {level}")
            
            self.save_cache()
            return lesson_data
        except Exception as e:
            print(f"Lỗi khi tạo bài học: {str(e)}")
            return None
    
    def _generate_course_templates(self, level, count=1):
        """Tạo mẫu khóa học theo cấp độ và lưu vào cache"""
        print(f"Đang tạo {count} mẫu khóa học cấp độ {level}...")
        if level not in self.course_cache:
            self.course_cache[level] = []
            
        for _ in range(count):
            try:
                course_data = self._generate_course_template(level)
                if course_data:
                    self.course_cache[level].append(course_data)
                    print(f"Đã tạo thành công mẫu khóa học cấp độ {level} #{len(self.course_cache[level])}")
            except Exception as e:
                print(f"Lỗi khi tạo mẫu khóa học cấp độ {level}: {str(e)}")
        
        # Giới hạn số lượng mẫu khóa học
        self.course_cache[level] = self.course_cache[level][-3:]  # Giữ tối đa 3 mẫu mỗi cấp độ
        self.save_cache()
    
    def _generate_course_template(self, level, weaknesses=None, user_id=None):
        """Tạo mẫu khóa học dựa trên cấp độ và điểm yếu"""
        print(f"Đang tạo mẫu khóa học cấp độ {level}...")
        
        # Tạo khóa học bằng AI
        try:
            # Tạo khóa học bằng AI với các thông tin bổ sung
            user_profile = None
            if user_id:
                # Giả lập tạo thông tin người dùng (trong thực tế sẽ lấy từ cơ sở dữ liệu)
                user_profile = {
                    'learning_goals': 'Nâng cao khả năng giao tiếp và hiểu biết văn hóa',
                    'preferred_topics': 'Travel, Business, Technology'
                }
            
            # Gọi AI để tạo khóa học
            course_data = self.ai.generate_course(level, weaknesses, user_profile)
            
            if course_data and 'title' in course_data:
                print(f"Đã tạo thành công khóa học: {course_data['title']}")
                return course_data
        except Exception as e:
            print(f"Lỗi khi tạo khóa học bằng AI: {str(e)}")
        
        # Phương pháp dự phòng nếu AI thất bại
        # Xác định topics phù hợp với level
        if level in ["Beginner (A1)", "Elementary (A2)"]:
            topics = ["Greetings", "Family", "Food", "Daily Activities"]
            title = f"Khóa học tiếng Anh cơ bản cho người mới bắt đầu ({level})"
            description = f"Khóa học này được tạo tự động dựa trên kết quả bài test. Thiết kế đặc biệt cho trình độ {level}, giúp bạn nắm vững các kiến thức cơ bản về tiếng Anh."
        elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
            topics = ["Travel", "Work", "Culture", "Media"]
            title = f"Tiếng Anh giao tiếp trung cấp ({level})"
            description = f"Khóa học được AI tạo riêng dựa vào kết quả bài kiểm tra. Tập trung nâng cao khả năng giao tiếp tiếng Anh cho công việc và đời sống hàng ngày ở trình độ {level}."
        else:  # C1, C2
            topics = ["Business", "Academic", "Literature", "Global Issues"]
            title = f"Tiếng Anh nâng cao - chuyên sâu ({level})"
            description = f"Khóa học chuyên sâu được cá nhân hóa theo trình độ {level}, giúp bạn làm chủ tiếng Anh ở cấp độ gần với người bản xứ."
        
        # Xác định chủ đề cho bài học
        topics_for_lessons = []
        if level in ["Beginner (A1)", "Elementary (A2)"]:
            topics_for_lessons = ["Greetings", "Family", "Food", "Travel", "Daily Activities"]
        elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
            topics_for_lessons = ["Work", "Hobbies", "Travel", "Culture", "Media"]
        else:  # C1, C2
            topics_for_lessons = ["Business", "Academic", "Literature", "Technology", "Global Issues"]
        
        # Tạo nội dung các bài học
        lessons = []
        for i, topic in enumerate(topics_for_lessons[:5], 1):
            weakness = weaknesses[i-1] if weaknesses and i <= len(weaknesses) else None
            
            # Tập trung kỹ năng dựa trên điểm yếu
            focus_skills = ["listening", "speaking", "reading", "writing"]
            if weakness:
                for skill in ["listening", "speaking", "reading", "writing"]:
                    if skill in weakness.lower():
                        # Đưa kỹ năng yếu lên đầu danh sách
                        focus_skills.remove(skill)
                        focus_skills.insert(0, skill)
            
            # Tạo nội dung bài học
            lesson = {
                "title": f"Bài {i}: {topic}",
                "order": i,
                "description": f"Bài học về {topic} tập trung vào kỹ năng {focus_skills[0]} và {focus_skills[1]}",
                "topic": topic,
                "estimated_time": 45,
                "focus_skills": focus_skills[:2],
                "objectives": [
                    f"Học từ vựng liên quan đến {topic}",
                    f"Luyện tập kỹ năng {focus_skills[0]} trong tình huống thực tế",
                    f"Phát triển kỹ năng {focus_skills[1]}"
                ],
                "weakness": weakness
            }
            lessons.append(lesson)
        
        # Tạo các bài kiểm tra định kỳ
        assessments = [
            {
                "title": "Kiểm tra giữa kỳ",
                "description": "Đánh giá tiến độ học tập sau 4 tuần đầu tiên",
                "week": 4,
                "type": "midterm",
                "focus_areas": topics_for_lessons[:3]
            },
            {
                "title": "Kiểm tra cuối kỳ",
                "description": "Đánh giá toàn diện các kiến thức và kỹ năng đã học",
                "week": 8,
                "type": "final",
                "focus_areas": topics_for_lessons
            }
        ]
        
        # Tạo các tài nguyên và hoạt động bổ sung
        recommendations = [
            {
                "type": "resource",
                "title": "Từ điển trực tuyến",
                "description": "Công cụ tra cứu từ vựng hữu ích",
                "url": "https://dictionary.cambridge.org/"
            },
            {
                "type": "activity",
                "title": "Luyện nghe với podcast",
                "description": "Nghe các podcast tiếng Anh để nâng cao kỹ năng nghe",
                "url": "#"
            },
            {
                "type": "resource",
                "title": f"Sách ngữ pháp tiếng Anh cho cấp độ {level}",
                "description": "Tài liệu tham khảo về ngữ pháp",
                "url": "#"
            }
        ]
        
        # Cấu trúc khóa học mẫu
        course_template = {
            "title": title,
            "description": description,
            "level": level,
            "is_published": True,
            "duration_weeks": 8,
            "topic": random.choice(topics),
            "is_auto_generated": True,
            "lessons": lessons,
            "assessments": assessments,
            "recommendations": recommendations,
            "template_created_at": datetime.utcnow().isoformat(),
        }
        
        if user_id:
            course_template["created_for"] = user_id
        
        return course_template
    
    def start_scheduler(self):
        """Khởi động scheduler để định kỳ làm mới nội dung"""
        if self.scheduler_running:
            print("Scheduler đã đang chạy!")
            return
        
        def scheduler_task():
            print("Scheduler đang chạy...")
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        
        # Hàm kiểm tra ngày đầu tiên của tháng
        def monthly_task():
            if datetime.utcnow().day == 1:
                print("Thực hiện cập nhật khóa học hàng tháng...")
                for level in self.ai.levels:
                    self._generate_course_templates(level, 1)
                
                # Cập nhật độ sáng tạo của AI
                self.ai.update_creativity_daily()
        
        # Hàng ngày, tạo thêm bài test xếp loại mới
        schedule.every().day.at("03:00").do(lambda: self._generate_placement_tests(1))
        
        # Hàng tuần vào thứ Hai, tạo thêm bài test các cấp độ mới
        schedule.every().monday.at("04:00").do(
            lambda: [self._generate_level_tests(level, 1) for level in self.ai.levels]
        )
        
        # Chạy hàng ngày lúc 5 giờ sáng, nhưng chỉ xử lý vào ngày 1 hàng tháng
        schedule.every().day.at("05:00").do(monthly_task)
        
        # Khởi động thread chạy scheduler
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=scheduler_task)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        print("Đã khởi động scheduler làm mới nội dung!")
    
    def stop_scheduler(self):
        """Dừng scheduler"""
        if self.scheduler_running:
            self.scheduler_running = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=2)
            print("Đã dừng scheduler!")

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'khoa_bi_mat_cua_ung_dung_hoc_tieng_anh'
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

# Khởi tạo AI
ai = DeepseekAI()

# Khởi tạo ContentCache với đối tượng AI
content_cache = ContentCache(ai)

# Định nghĩa các mô hình dữ liệu
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    fullname = db.Column(db.String(120))
    role = db.Column(db.String(20), default='student')  # 'admin', 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    avatar = db.Column(db.String(255), default='default_avatar.png')
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    
    # Quan hệ
    profile = db.relationship('UserProfile', backref='user', uselist=False)
    test_results = db.relationship('TestResult', backref='user')
    course_enrollments = db.relationship('CourseEnrollment', backref='user')
    notifications = db.relationship('Notification', backref='user')
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_published = db.Column(db.Boolean, default=True)  # Mặc định là published
    image = db.Column(db.String(255), default='default_course.jpg')
    duration_weeks = db.Column(db.Integer)
    topic = db.Column(db.String(100))
    is_auto_generated = db.Column(db.Boolean, default=True)  # Mặc định là tự động tạo
    
    # Quan hệ
    lessons = db.relationship('Lesson', backref='course', cascade="all, delete-orphan")
    enrollments = db.relationship('CourseEnrollment', backref='course', cascade="all, delete-orphan")
    
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
    topic = db.Column(db.String(100))  # Chủ đề của bài học
    focus_skills = db.Column(db.String(255))  # JSON string of skills (listening, speaking, reading, writing)
    
    # Quan hệ
    exercises = db.relationship('Exercise', backref='lesson', cascade="all, delete-orphan")
    completion_records = db.relationship('LessonCompletion', backref='lesson', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Lesson {self.title}>'

class LessonCompletion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)
    score = db.Column(db.Float, default=0.0)  # Điểm đánh giá (0-10)
    time_spent = db.Column(db.Integer)  # Seconds
    notes = db.Column(db.Text)  # Ghi chú của người dùng
    
    # Quan hệ
    user = db.relationship('User', backref='lesson_completions')
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'lesson_id', name='uix_user_lesson'),
    )
    
    def __repr__(self):
        return f'<LessonCompletion for User #{self.user_id} Lesson #{self.lesson_id}>'

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
    skill = db.Column(db.String(50))  # listening, speaking, reading, writing
    
    # Quan hệ
    submissions = db.relationship('ExerciseSubmission', backref='exercise', cascade="all, delete-orphan")
    
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
    level = db.Column(db.String(50))  # A1, A2, B1, B2, C1, C2
    questions = db.Column(db.Text)  # JSON format
    time_limit = db.Column(db.Integer)  # Minutes
    passing_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    instructions = db.Column(db.Text)  # Hướng dẫn làm bài
    
    # Quan hệ
    results = db.relationship('TestResult', backref='test', cascade="all, delete-orphan")
    
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
    skill_scores = db.Column(db.Text)  # JSON format for scores in different skills
    
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

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text)
    type = db.Column(db.String(50))  # system, course, test, etc.
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

class Statistic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    study_time = db.Column(db.Integer, default=0)  # Minutes
    exercises_completed = db.Column(db.Integer, default=0)
    tests_completed = db.Column(db.Integer, default=0)
    points_earned = db.Column(db.Integer, default=0)
    lessons_completed = db.Column(db.Integer, default=0)
    
    # Quan hệ
    user = db.relationship('User', backref='statistics')
    
    def __repr__(self):
        return f'<Statistic for User #{self.user_id} on {self.date}>'

class StudyPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    goal = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='active')  # active, completed, cancelled
    
    # Quan hệ
    user = db.relationship('User', backref='study_plans')
    tasks = db.relationship('StudyTask', backref='plan', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<StudyPlan {self.title} for User #{self.user_id}>'

class StudyTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    due_date = db.Column(db.Date)
    is_completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)
    priority = db.Column(db.Integer, default=1)  # 1-5 (1: highest, 5: lowest)
    
    def __repr__(self):
        return f'<StudyTask {self.title} for Plan #{self.plan_id}>'

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_type = db.Column(db.String(50))  # course, lesson, test, exercise, etc.
    content_id = db.Column(db.Integer)
    rating = db.Column(db.Integer)  # 1-5
    comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Quan hệ
    user = db.relationship('User', backref='feedbacks')
    
    def __repr__(self):
        return f'<Feedback by User #{self.user_id} for {self.content_type} #{self.content_id}>'

class Discussion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    topic = db.Column(db.String(100))
    is_pinned = db.Column(db.Boolean, default=False)
    
    # Quan hệ
    user = db.relationship('User', backref='discussions')
    comments = db.relationship('DiscussionComment', backref='discussion', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Discussion {self.title}>'

class DiscussionComment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussion.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    # Quan hệ
    user = db.relationship('User', backref='discussion_comments')
    
    def __repr__(self):
        return f'<DiscussionComment by User #{self.user_id}>'

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
        
        # Tạo thông báo chào mừng
        notification = Notification(
            user_id=user.id,
            title="Chào mừng đến với ứng dụng học tiếng Anh!",
            content="Chào mừng bạn đến với nền tảng học tiếng Anh thông minh. Làm bài kiểm tra xếp loại để bắt đầu hành trình học tập của bạn.",
            type="system",
            url=url_for('dashboard')
        )
        db.session.add(notification)
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
        recommended_courses = Course.query.filter_by(level=user_profile.language_level, is_published=True).limit(3).all()
    else:
        # Nếu chưa có level, gợi ý khóa học cho người mới bắt đầu
        recommended_courses = Course.query.filter_by(level="Beginner (A1)", is_published=True).limit(3).all()
    
    # Kiểm tra xem đã làm bài test xếp loại chưa
    has_placement_test = TestResult.query.join(Test).filter(
        TestResult.user_id == current_user.id,
        Test.type == 'placement'
    ).first()
    
    # Lấy thống kê học tập
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    stats = Statistic.query.filter(
        Statistic.user_id == current_user.id,
        Statistic.date >= week_ago
    ).order_by(Statistic.date).all()
    
    # Tạo dữ liệu biểu đồ tiến độ
    chart_data = {
        'labels': [],
        'values': []
    }
    
    for i in range(7):
        date = week_ago + timedelta(days=i)
        date_str = date.strftime('%d/%m')
        chart_data['labels'].append(date_str)
        
        # Tìm thống kê cho ngày này
        day_stat = next((s for s in stats if s.date == date), None)
        if day_stat:
            # Tính điểm hoạt động (tối đa 100)
            activity_score = min(100, day_stat.points_earned + day_stat.exercises_completed * 5 + day_stat.study_time // 10)
            chart_data['values'].append(activity_score)
        else:
            chart_data['values'].append(0)
    
    # Tạo đường dẫn biểu đồ
    chart_path = os.path.join('static', 'images', 'charts', f'progress_{current_user.id}.png')
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    
    TienIchMedia.tao_bieu_do_tien_bo(chart_data, chart_path)
    
    return render_template('dashboard.html', 
                          user_profile=user_profile, 
                          courses=courses, 
                          recent_tests=recent_tests, 
                          notifications=notifications, 
                          recommended_courses=recommended_courses,
                          has_placement_test=has_placement_test,
                          chart_path=chart_path,
                          stats=stats)

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
    user_badges = [Badge.query.get(b.badge_id) for b in badges]
    
    # Lấy thống kê học tập
    total_study_time = db.session.query(db.func.sum(Statistic.study_time)).filter_by(user_id=current_user.id).scalar() or 0
    total_exercises = db.session.query(db.func.sum(Statistic.exercises_completed)).filter_by(user_id=current_user.id).scalar() or 0
    total_tests = db.session.query(db.func.sum(Statistic.tests_completed)).filter_by(user_id=current_user.id).scalar() or 0
    
    return render_template('profile.html', 
                          profile=profile, 
                          badges=user_badges, 
                          total_study_time=total_study_time,
                          total_exercises=total_exercises,
                          total_tests=total_tests)

@app.route('/test/placement')
@login_required
def placement_test():
    # Lấy bài test xếp loại có sẵn hoặc tạo mới nếu cần
    placement_test = Test.query.filter_by(type='placement', is_active=True).first()
    
    if not placement_test:
        try:
            # Lấy từ cache hoặc tạo mới nếu cache trống
            test_data = content_cache.get_placement_test()
            
            placement_test = Test(
                title="Bài test xếp loại",
                description="Bài test này sẽ giúp xác định trình độ tiếng Anh của bạn.",
                type="placement",
                questions=json.dumps(test_data["questions"]),
                time_limit=test_data["time_limit"] // 60,  # Chuyển đổi giây sang phút
                passing_score=test_data["passing_score"],
                is_active=True,
                instructions=test_data.get("instructions", "Hãy đọc kỹ và trả lời các câu hỏi trong thời gian quy định.")
            )
            db.session.add(placement_test)
            db.session.commit()
        except Exception as e:
            # Nếu có lỗi khi tạo từ cache, sử dụng AI trực tiếp
            print(f"Lỗi khi lấy test từ cache: {str(e)}")
            print("Sử dụng AI để tạo trực tiếp...")
            
            test_data = ai.generate_test()
            placement_test = Test(
                title="Bài test xếp loại",
                description="Bài test này sẽ giúp xác định trình độ tiếng Anh của bạn.",
                type="placement",
                questions=json.dumps(test_data["questions"]),
                time_limit=test_data["time_limit"] // 60,
                passing_score=test_data["passing_score"],
                is_active=True,
                instructions=test_data.get("instructions", "Hãy đọc kỹ và trả lời các câu hỏi trong thời gian quy định.")
            )
            db.session.add(placement_test)
            db.session.commit()
    
    # Kiểm tra xem người dùng đã làm bài test xếp loại chưa
    existing_result = TestResult.query.filter_by(
        user_id=current_user.id,
        test_id=placement_test.id
    ).first()
    
    if existing_result:
        flash('Bạn đã hoàn thành bài test xếp loại. Bạn có thể xem kết quả hoặc làm lại bài test.', 'info')
        return redirect(url_for('test_result', test_id=placement_test.id))
    
    # Lấy danh sách câu hỏi
    questions = json.loads(placement_test.questions)
    
    return render_template('take_test.html', test=placement_test, questions=questions, time_limit=placement_test.time_limit * 60)

@app.route('/test/<int:test_id>')
@login_required
def take_test(test_id):
    test = Test.query.get_or_404(test_id)
    questions = json.loads(test.questions)
    
    return render_template('take_test.html', test=test, questions=questions, time_limit=test.time_limit * 60)

# Hàm tạo khóa học tự động dựa trên trình độ
def create_auto_course(level, user_id, weaknesses=None):
    """Tạo khóa học phù hợp với trình độ người dùng sau khi hoàn thành bài test xếp loại"""
    print(f"Bắt đầu tạo khóa học tự động cho cấp độ {level}, user ID {user_id}")
    
    # Biến lưu trữ thông tin khóa học mẫu
    course_template = None
    
    # Bước 1: Thử lấy mẫu khóa học từ cache
    try:
        print(f"Đang lấy mẫu khóa học từ cache cho cấp độ {level}...")
        course_template = content_cache.get_course(level, user_id, weaknesses)
        print("Đã lấy thành công mẫu khóa học từ cache")
    except Exception as e:
        print(f"Lỗi khi lấy mẫu khóa học từ cache: {str(e)}")
        course_template = None

    # Bước 2: Nếu không lấy được từ cache, xác định thông tin khóa học theo cách thủ công
    if course_template is None:
        print("Tạo thông tin khóa học theo cách thủ công...")
        
        # Xác định topics và thông tin cơ bản phù hợp với level
        if level in ["Beginner (A1)", "Elementary (A2)"]:
            topics = ["Greetings", "Family", "Food", "Daily Activities"]
            title = f"Khóa học tiếng Anh cơ bản cho người mới bắt đầu ({level})"
            description = f"Khóa học này được tạo tự động dựa trên kết quả bài test của bạn. Thiết kế đặc biệt cho trình độ {level}, giúp bạn nắm vững các kiến thức cơ bản về tiếng Anh."
        elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
            topics = ["Travel", "Work", "Culture", "Media"]
            title = f"Tiếng Anh giao tiếp trung cấp ({level})"
            description = f"Khóa học được AI tạo riêng cho bạn dựa vào kết quả bài kiểm tra. Tập trung nâng cao khả năng giao tiếp tiếng Anh cho công việc và đời sống hàng ngày ở trình độ {level}."
        else:  # C1, C2
            topics = ["Business", "Academic", "Literature", "Global Issues"]
            title = f"Tiếng Anh nâng cao - chuyên sâu ({level})"
            description = f"Khóa học chuyên sâu được cá nhân hóa theo trình độ {level} của bạn, giúp bạn làm chủ tiếng Anh ở cấp độ gần với người bản xứ."
        
        # Tạo mẫu khóa học thủ công
        selected_topic = random.choice(topics)
        course_template = {
            "title": title,
            "description": description,
            "level": level,
            "topic": selected_topic,
            "duration_weeks": 8,
            "is_auto_generated": True
        }
        
        # Xác định chủ đề cho các bài học
        topics_for_lessons = []
        if level in ["Beginner (A1)", "Elementary (A2)"]:
            topics_for_lessons = ["Greetings", "Family", "Food", "Travel", "Daily Activities"]
        elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
            topics_for_lessons = ["Work", "Hobbies", "Travel", "Culture", "Media"]
        else:  # C1, C2
            topics_for_lessons = ["Business", "Academic", "Literature", "Technology", "Global Issues"]
        
        # Tạo thông tin cho các bài học
        lessons = []
        for i, topic in enumerate(topics_for_lessons[:5], 1):
            weakness = weaknesses[i-1] if weaknesses and i <= len(weaknesses) else None
            
            lesson = {
                "title": f"Bài {i}: {topic}",
                "order": i,
                "estimated_time": 45,
                "topic": topic,
                "weakness": weakness
            }
            lessons.append(lesson)
        
        course_template["lessons"] = lessons

    # Bước 3: Tạo khóa học với thông tin từ template
    try:
        print(f"Tạo khóa học mới: {course_template['title']}")
        course = Course(
            title=course_template["title"],
            description=course_template["description"],
            level=level,
            is_published=True,
            duration_weeks=course_template.get("duration_weeks", 8),
            topic=course_template["topic"],
            is_auto_generated=True
        )
        db.session.add(course)
        db.session.commit()
        print(f"Đã tạo khóa học ID: {course.id}")
    except Exception as e:
        print(f"Lỗi khi tạo khóa học: {str(e)}")
        db.session.rollback()
        # Tạo khóa học với thông tin tối thiểu
        course = Course(
            title=f"Khóa học tiếng Anh {level}",
            description=f"Khóa học cho cấp độ {level}",
            level=level,
            is_published=True,
            duration_weeks=8,
            topic="General English",
            is_auto_generated=True
        )
        db.session.add(course)
        db.session.commit()
        print(f"Đã tạo khóa học dự phòng ID: {course.id}")

    # Bước 4: Tạo các bài học và bài tập
    lessons_data = course_template.get("lessons", [])
    if not lessons_data:
        # Tạo ít nhất 3 bài học nếu không có trong template
        topics_default = ["Vocabulary", "Grammar", "Communication", "Reading", "Listening"]
        lessons_data = [
            {"title": f"Bài {i}: {topic}", "order": i, "estimated_time": 45, "topic": topic}
            for i, topic in enumerate(topics_default[:5], 1)
        ]
    
    # Tạo các bài học
    for lesson_data in lessons_data:
        try:
            print(f"Tạo bài học: {lesson_data['title']}")
            # Thử lấy nội dung bài học từ cache
            try:
                lesson_content = content_cache.get_lesson(level, lesson_data.get("topic", "General"), lesson_data.get("weakness"))
            except Exception as e:
                print(f"Lỗi khi lấy nội dung bài học từ cache: {str(e)}")
                # Nếu lỗi, tạo trực tiếp bằng AI
                try:
                    lesson_content = ai.generate_lesson(level, lesson_data.get("topic", "General"), lesson_data.get("weakness"))
                except Exception as e2:
                    print(f"Lỗi khi tạo nội dung bài học bằng AI: {str(e2)}")
                    # Nếu vẫn lỗi, tạo nội dung giả
                    lesson_content = {
                        "title": lesson_data['title'],
                        "description": f"Bài học về {lesson_data.get('topic', 'General')} cho cấp độ {level}",
                        "objectives": ["Hiểu kiến thức cơ bản", "Thực hành trong tình huống thực tế"],
                        "sections": [
                            {"title": "Giới thiệu", "content": f"<p>Nội dung bài học {lesson_data['title']}</p>"}
                        ]
                    }
            
            # Xác định các kỹ năng trọng tâm
            focus_skills = ["listening", "speaking", "reading", "writing"]
            if lesson_data.get("weakness"):
                for skill in focus_skills.copy():
                    if skill in lesson_data["weakness"].lower():
                        focus_skills.remove(skill)
                        focus_skills.insert(0, skill)
            
            # Tạo bài học
            lesson = Lesson(
                course_id=course.id,
                title=lesson_data['title'],
                content=json.dumps(lesson_content),
                order=lesson_data.get('order', 1),
                estimated_time=lesson_data.get('estimated_time', 45),
                topic=lesson_data.get('topic', 'General'),
                focus_skills=json.dumps(focus_skills[:2])
            )
            db.session.add(lesson)
            db.session.commit()
            print(f"Đã tạo bài học ID: {lesson.id}")
            
            # Tạo bài tập cho bài học
            try:
                print(f"Tạo bài tập cho bài học: {lesson.id}")
                # Thử lấy bài tập từ cache
                try:
                    exercise_data = content_cache.get_topic_test(level, lesson_data.get("topic", "General"), 5)
                except Exception as e:
                    print(f"Lỗi khi lấy bài tập từ cache: {str(e)}")
                    # Nếu lỗi, tạo trực tiếp bằng AI
                    try:
                        exercise_data = ai.generate_test(level, lesson_data.get("topic", "General"), 5)
                    except Exception as e2:
                        print(f"Lỗi khi tạo bài tập bằng AI: {str(e2)}")
                        # Nếu vẫn lỗi, tạo nội dung giả
                        exercise_data = {
                            "questions": [
                                {
                                    "id": f"q{i+1}",
                                    "type": "multiple_choice",
                                    "question": f"Câu hỏi mẫu {i+1} về {lesson_data.get('topic', 'General')}?",
                                    "options": ["Option A", "Option B", "Option C", "Option D"],
                                    "difficulty": level
                                } for i in range(5)
                            ]
                        }
                
                # Tạo bài tập listening
                listening_exercise = Exercise(
                    lesson_id=lesson.id,
                    title=f"Bài tập nghe: {lesson_data.get('topic', 'General')}",
                    description=f"Luyện tập kỹ năng nghe với chủ đề {lesson_data.get('topic', 'General')}",
                    type="listening",
                    content=json.dumps([q for q in exercise_data.get("questions", []) if q.get("type") == "listening"] or []),
                    time_limit=10,
                    points=10,
                    skill="listening"
                )
                db.session.add(listening_exercise)
                
                # Tạo bài tập vocabulary và grammar
                vocab_exercise = Exercise(
                    lesson_id=lesson.id,
                    title=f"Từ vựng và ngữ pháp: {lesson_data.get('topic', 'General')}",
                    description=f"Luyện tập từ vựng và ngữ pháp với chủ đề {lesson_data.get('topic', 'General')}",
                    type="quiz",
                    content=json.dumps([q for q in exercise_data.get("questions", []) if q.get("type") in ["grammar", "vocabulary", "multiple_choice"]] or []),
                    time_limit=15,
                    points=10,
                    skill="vocabulary_grammar"
                )
                db.session.add(vocab_exercise)
                
                # Tạo bài tập reading
                reading_exercise = Exercise(
                    lesson_id=lesson.id,
                    title=f"Bài tập đọc hiểu: {lesson_data.get('topic', 'General')}",
                    description=f"Luyện tập kỹ năng đọc hiểu với chủ đề {lesson_data.get('topic', 'General')}",
                    type="reading",
                    content=json.dumps([q for q in exercise_data.get("questions", []) if q.get("type") == "reading"] or []),
                    time_limit=15,
                    points=10,
                    skill="reading"
                )
                db.session.add(reading_exercise)
                
                # Tạo bài tập speaking and writing
                speaking_writing_exercise = Exercise(
                    lesson_id=lesson.id,
                    title=f"Nói và viết: {lesson_data.get('topic', 'General')}",
                    description=f"Luyện tập kỹ năng nói và viết với chủ đề {lesson_data.get('topic', 'General')}",
                    type="speaking_writing",
                    content=json.dumps([q for q in exercise_data.get("questions", []) if q.get("type") in ["speaking", "writing"]] or []),
                    time_limit=20,
                    points=15,
                    skill="speaking_writing"
                )
                db.session.add(speaking_writing_exercise)
                
                db.session.commit()
                print(f"Đã tạo các bài tập cho bài học ID: {lesson.id}")
            except Exception as e:
                print(f"Lỗi khi tạo bài tập: {str(e)}")
                db.session.rollback()
                # Tiếp tục với bài học tiếp theo
        except Exception as e:
            print(f"Lỗi khi tạo bài học: {str(e)}")
            db.session.rollback()
            # Tiếp tục với bài học tiếp theo

    # Bước 5: Tự động đăng ký người dùng vào khóa học
    try:
        print(f"Đăng ký người dùng ID {user_id} vào khóa học ID {course.id}")
        enrollment = CourseEnrollment(
            user_id=user_id,
            course_id=course.id,
            last_accessed=datetime.utcnow()
        )
        db.session.add(enrollment)
        db.session.commit()
        print("Đã đăng ký thành công")
    except Exception as e:
        print(f"Lỗi khi đăng ký người dùng vào khóa học: {str(e)}")
        db.session.rollback()
        # Tiếp tục, không dừng lại

    # Bước 6: Tạo bài test cuối khóa
    try:
        print("Tạo bài test cuối khóa")
        # Thử lấy bài test từ cache
        try:
            test_data = content_cache.get_level_test(level)
        except Exception as e:
            print(f"Lỗi khi lấy bài test từ cache: {str(e)}")
            # Nếu lỗi, tạo trực tiếp bằng AI
            try:
                test_data = ai.generate_test(level)
            except Exception as e2:
                print(f"Lỗi khi tạo bài test bằng AI: {str(e2)}")
                # Nếu vẫn lỗi, tạo nội dung giả
                test_data = {
                    "questions": [
                        {
                            "id": f"q{i+1}",
                            "type": "multiple_choice",
                            "question": f"Câu hỏi test cuối khóa {i+1}?",
                            "options": ["Option A", "Option B", "Option C", "Option D"],
                            "difficulty": level
                        } for i in range(10)
                    ],
                    "time_limit": 1800,  # 30 phút
                    "passing_score": 7.0
                }
        
        # Tạo bài test
        test = Test(
            title=f"Bài kiểm tra cuối khóa - {level}",
            description=f"Đánh giá kiến thức sau khi hoàn thành khóa học {level}",
            type="course",
            level=level,
            questions=json.dumps(test_data.get("questions", [])),
            time_limit=30,
            passing_score=7.0,
            is_active=True
        )
        db.session.add(test)
        db.session.commit()
        print(f"Đã tạo bài test cuối khóa ID: {test.id}")
    except Exception as e:
        print(f"Lỗi khi tạo bài test cuối khóa: {str(e)}")
        db.session.rollback()
        # Tiếp tục, không dừng lại
    
    print(f"Đã hoàn thành tạo khóa học tự động ID: {course.id}")
    return course

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
    
    # Cập nhật thống kê
    today = datetime.utcnow().date()
    stat = Statistic.query.filter_by(user_id=current_user.id, date=today).first()
    if not stat:
        stat = Statistic(user_id=current_user.id, date=today)
        db.session.add(stat)
    
    stat.tests_completed += 1
    stat.points_earned += int(feedback["score"] * 10)  # Thêm điểm dựa vào kết quả
    
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
    
    # Nếu là bài test xếp loại, tạo khóa học tự động và đăng ký người dùng
    if test.type == 'placement':
        weaknesses = feedback.get("weaknesses", [])
        auto_course = create_auto_course(feedback["level"], current_user.id, weaknesses)
        
        # Tạo thông báo về khóa học mới
        notification = Notification(
            user_id=current_user.id,
            title="Khóa học được tạo tự động",
            content=f"Dựa trên kết quả bài test xếp loại, hệ thống đã tạo và đăng ký bạn vào khóa học '{auto_course.title}'.",
            type="course",
            url=url_for('course_detail', course_id=auto_course.id)
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
    beginner_courses = [c for c in published_courses if c.level and c.level.startswith(('Beginner', 'Elementary'))]
    intermediate_courses = [c for c in published_courses if c.level and c.level.startswith(('Inter'))]
    advanced_courses = [c for c in published_courses if c.level and c.level.startswith(('Advanced', 'Proficient'))]
    
    # Tìm khóa học đề xuất
    user_profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    recommended_courses = []
    
    if user_profile and user_profile.language_level:
        # Tạo danh sách khóa học đề xuất dựa trên cấp độ và chủ đề yêu thích
        preferred_topics = user_profile.preferred_topics.split(',') if user_profile.preferred_topics else []
        
        # Khóa học cùng cấp độ
        level_courses = [c for c in published_courses if c.level == user_profile.language_level and c.id not in enrolled_course_ids]
        
        # Lọc theo chủ đề yêu thích nếu có
        if preferred_topics:
            for topic in preferred_topics:
                topic_courses = [c for c in level_courses if c.topic and topic.strip().lower() in c.topic.lower()]
                recommended_courses.extend(topic_courses)
        
        # Nếu không có khóa học phù hợp với chủ đề, lấy tất cả khóa học cùng cấp độ
        if not recommended_courses:
            recommended_courses = level_courses
        
        # Giới hạn số lượng đề xuất
        recommended_courses = recommended_courses[:3]
    
    return render_template('courses.html', 
                          beginner_courses=beginner_courses,
                          intermediate_courses=intermediate_courses,
                          advanced_courses=advanced_courses,
                          enrolled_course_ids=enrolled_course_ids,
                          recommended_courses=recommended_courses)

@app.route('/course/<int:course_id>')
@login_required
def course_detail(course_id):
    course = Course.query.get_or_404(course_id)
    lessons = Lesson.query.filter_by(course_id=course.id).order_by(Lesson.order).all()
    
    # Kiểm tra người dùng đã đăng ký khóa học chưa
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=course.id).first()
    
    # Tính toán tiến độ học tập nếu đã đăng ký
    lesson_progress = {}
    if enrollment:
        for lesson in lessons:
            # Kiểm tra hoàn thành bài học
            completion = LessonCompletion.query.filter_by(user_id=current_user.id, lesson_id=lesson.id).first()
            if completion:
                lesson_progress[lesson.id] = {
                    'completed': True,
                    'score': completion.score,
                    'completed_at': completion.completed_at
                }
            else:
                lesson_progress[lesson.id] = {
                    'completed': False
                }
    
    return render_template('course_detail.html', 
                          course=course, 
                          lessons=lessons, 
                          enrollment=enrollment,
                          lesson_progress=lesson_progress)

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
    
    # Lấy bài tập
    exercises = Exercise.query.filter_by(lesson_id=lesson.id).all()
    
    # Lấy thông tin hoàn thành
    completion = LessonCompletion.query.filter_by(user_id=current_user.id, lesson_id=lesson.id).first()
    
    # Nếu nội dung bài học là JSON, parse nó
    try:
        content = json.loads(lesson.content)
    except:
        content = {"html": lesson.content}  # Nếu không phải JSON, coi như HTML
    
    # Xử lý media trong nội dung bài học
    if "sections" in content:
        for section in content["sections"]:
            # Xử lý audio nếu có
            if "listening_activities" in section:
                for activity in section["listening_activities"]:
                    if "audio_url" in activity:
                        # Đường dẫn audio đã có
                        pass
                    elif "audio_text" in activity:
                        # Tạo audio mới từ text
                        audio_text = activity["audio_text"]
                        
                        # Tạo thư mục nếu chưa có
                        audio_dir = os.path.join('static', 'audio', 'lessons', f"lesson_{lesson.id}")
                        os.makedirs(audio_dir, exist_ok=True)
                        
                        # Tạo đường dẫn audio
                        audio_path = os.path.join(audio_dir, f"activity_{hash(audio_text) % 10000:04d}.mp3")
                        
                        # Kiểm tra nếu file chưa tồn tại
                        if not os.path.exists(audio_path):
                            TienIchMedia.tao_audio_tu_van_ban(audio_text, audio_path)
                        
                        # Thêm vào activity
                        activity["audio_url"] = audio_path
    
    # Cập nhật thời gian học
    update_study_time(current_user.id, 10)  # Mặc định tính 10 phút cho mỗi lần xem bài học
    
    return render_template('lesson_detail.html', 
                          lesson=lesson, 
                          course=course, 
                          exercises=exercises, 
                          content=content,
                          completion=completion)

@app.route('/lesson/<int:lesson_id>/complete', methods=['POST'])
@login_required
def complete_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    
    # Kiểm tra đăng ký khóa học
    course = Course.query.get(lesson.course_id)
    enrollment = CourseEnrollment.query.filter_by(user_id=current_user.id, course_id=course.id).first()
    if not enrollment:
        flash('Bạn chưa đăng ký khóa học này.', 'danger')
        return redirect(url_for('course_detail', course_id=course.id))
    
    # Kiểm tra nếu đã hoàn thành bài học trước đó
    existing_completion = LessonCompletion.query.filter_by(user_id=current_user.id, lesson_id=lesson.id).first()
    
    if existing_completion:
        # Cập nhật completion
        existing_completion.score = float(request.form.get('self_assessment', 8.0))
        existing_completion.notes = request.form.get('notes', '')
        existing_completion.completed_at = datetime.utcnow()
    else:
        # Tạo completion mới
        completion = LessonCompletion(
            user_id=current_user.id,
            lesson_id=lesson.id,
            score=float(request.form.get('self_assessment', 8.0)),
            notes=request.form.get('notes', ''),
            time_spent=int(request.form.get('time_spent', 600))  # Mặc định 10 phút
        )
        db.session.add(completion)
        
        # Cập nhật thống kê
        today = datetime.utcnow().date()
        stat = Statistic.query.filter_by(user_id=current_user.id, date=today).first()
        if not stat:
            stat = Statistic(user_id=current_user.id, date=today)
            db.session.add(stat)
        
        stat.lessons_completed += 1
        stat.points_earned += 20  # Điểm cho việc hoàn thành bài học
    
    # Cập nhật tiến độ khóa học
    total_lessons = Lesson.query.filter_by(course_id=course.id).count()
    completed_lessons = LessonCompletion.query.join(Lesson).filter(
        LessonCompletion.user_id == current_user.id,
        Lesson.course_id == course.id
    ).count()
    
    enrollment.progress = (completed_lessons / total_lessons) * 100 if total_lessons > 0 else 0
    
    # Kiểm tra nếu đã hoàn thành tất cả bài học
    if completed_lessons == total_lessons:
        enrollment.status = 'completed'
        
        # Tạo thông báo khi hoàn thành khóa học
        notification = Notification(
            user_id=current_user.id,
            title=f"Chúc mừng! Bạn đã hoàn thành khóa học",
            content=f"Bạn đã hoàn thành khóa học '{course.title}'. Hãy làm bài kiểm tra cuối khóa để đánh giá kết quả học tập.",
            type="course",
            url=url_for('course_detail', course_id=course.id)
        )
        db.session.add(notification)
    
    db.session.commit()
    
    # Kiểm tra huy hiệu
    check_and_award_badges(current_user.id)
    
    flash('Bạn đã hoàn thành bài học thành công!', 'success')
    return redirect(url_for('course_detail', course_id=course.id))

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
    
    try:
        # Parse nội dung bài tập
        content = json.loads(exercise.content)
    except:
        # Nếu không phải JSON, sử dụng nội dung thô
        content = []
    
    # Xử lý media trong bài tập
    for question in content:
        # Xử lý audio cho câu hỏi nghe
        if question.get('type') == 'listening' and 'audio_text' in question and not 'audio_url' in question:
            audio_text = question['audio_text']
            
            # Tạo thư mục nếu chưa có
            audio_dir = os.path.join('static', 'audio', 'exercises', f"exercise_{exercise.id}")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Tạo đường dẫn audio
            audio_path = os.path.join(audio_dir, f"question_{hash(audio_text) % 10000:04d}.mp3")
            
            # Kiểm tra nếu file chưa tồn tại
            if not os.path.exists(audio_path):
                TienIchMedia.tao_audio_tu_van_ban(audio_text, audio_path)
            
            # Thêm vào activity
            question["audio_url"] = audio_path
        
        # Xử lý hình ảnh cho câu hỏi đọc hiểu
        if question.get('type') == 'reading' and 'text' in question and len(question['text']) > 100 and not 'image_url' in question:
            text = question['text']
            
            # Tạo thư mục nếu chưa có
            images_dir = os.path.join('static', 'images', 'exercises', f"exercise_{exercise.id}")
            os.makedirs(images_dir, exist_ok=True)
            
            # Tạo đường dẫn hình ảnh
            image_path = os.path.join(images_dir, f"reading_{hash(text) % 10000:04d}.png")
            
            # Kiểm tra nếu file chưa tồn tại
            if not os.path.exists(image_path):
                TienIchMedia.tao_hinh_anh_van_ban(text, image_path, tieu_de="Reading Exercise")
            
            # Thêm vào question
            question["image_url"] = image_path
    
    return render_template('take_exercise.html', 
                          exercise=exercise, 
                          lesson=lesson, 
                          content=content)

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
    
    # Lấy đáp án đúng từ nội dung bài tập
    try:
        questions = json.loads(exercise.content)
        correct_answers = []
        for question in questions:
            if 'correct_answer' in question:
                correct_answers.append(question['correct_answer'])
            else:
                correct_answers.append('')  # Không có đáp án cụ thể (như cho câu hỏi viết)
    except:
        correct_answers = [""] * len(answers)  # Không có đáp án cụ thể
    
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
        # Cập nhật thời gian truy cập
        enrollment.last_accessed = datetime.utcnow()
    
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
    total_lessons = sum([s.lessons_completed for s in stats])
    
    # Lấy kết quả bài test gần đây
    recent_tests = TestResult.query.filter_by(user_id=current_user.id).order_by(TestResult.completed_at.desc()).limit(5).all()
    
    # Lấy bài tập đã hoàn thành gần đây
    recent_exercises = ExerciseSubmission.query.filter_by(user_id=current_user.id).order_by(ExerciseSubmission.submitted_at.desc()).limit(5).all()
    
    # Tạo dữ liệu biểu đồ
    chart_data = {
        'labels': [],
        'values': []
    }
    
    for i in range(7):
        date = week_ago + timedelta(days=i)
        date_str = date.strftime('%d/%m')
        chart_data['labels'].append(date_str)
        
        day_stat = next((s for s in stats if s.date == date), None)
        if day_stat:
            chart_data['values'].append(day_stat.points_earned)
        else:
            chart_data['values'].append(0)
    
    # Tạo đường dẫn biểu đồ
    chart_path = os.path.join('static', 'images', 'charts', f'stats_{current_user.id}.png')
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    
    TienIchMedia.tao_bieu_do_tien_bo(chart_data, chart_path, tieu_de="Điểm số trong 7 ngày qua")
    
    return render_template('statistics.html', 
                          stats=stats, 
                          total_study_time=total_study_time,
                          total_exercises=total_exercises,
                          total_tests=total_tests,
                          total_lessons=total_lessons,
                          total_points=total_points,
                          recent_tests=recent_tests,
                          recent_exercises=recent_exercises,
                          chart_path=chart_path)

@app.route('/study-plan', methods=['GET', 'POST'])
@login_required
def study_plan():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        goal = request.form.get('goal')
        start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d').date()
        
        # Tạo kế hoạch học tập mới
        plan = StudyPlan(
            user_id=current_user.id,
            title=title,
            description=description,
            goal=goal,
            start_date=start_date,
            end_date=end_date,
            status='active'
        )
        db.session.add(plan)
        db.session.commit()
        
        # Tạo các nhiệm vụ từ form
        task_titles = request.form.getlist('task_title[]')
        task_descriptions = request.form.getlist('task_description[]')
        task_due_dates = request.form.getlist('task_due_date[]')
        task_priorities = request.form.getlist('task_priority[]')
        
        for i in range(len(task_titles)):
            if task_titles[i].strip():  # Kiểm tra tiêu đề không trống
                task = StudyTask(
                    plan_id=plan.id,
                    title=task_titles[i],
                    description=task_descriptions[i] if i < len(task_descriptions) else "",
                    due_date=datetime.strptime(task_due_dates[i], '%Y-%m-%d').date() if i < len(task_due_dates) and task_due_dates[i] else None,
                    priority=int(task_priorities[i]) if i < len(task_priorities) and task_priorities[i] else 3
                )
                db.session.add(task)
        
        db.session.commit()
        flash('Kế hoạch học tập đã được tạo thành công!', 'success')
        return redirect(url_for('study_plan'))
    
    # Lấy tất cả kế hoạch học tập của người dùng
    plans = StudyPlan.query.filter_by(user_id=current_user.id).order_by(StudyPlan.created_at.desc()).all()
    
    # Tạo kế hoạch tự động từ AI nếu yêu cầu
    auto_plan = None
    if request.args.get('auto'):
        profile = UserProfile.query.filter_by(user_id=current_user.id).first()
        if profile and profile.language_level:
            # Tạo kế hoạch học tập tự động dựa trên cấp độ
            auto_plan = generate_auto_study_plan(profile.language_level)
    
    return render_template('study_plan.html', plans=plans, auto_plan=auto_plan)

@app.route('/study-plan/<int:plan_id>')
@login_required
def view_study_plan(plan_id):
    plan = StudyPlan.query.filter_by(id=plan_id, user_id=current_user.id).first_or_404()
    tasks = StudyTask.query.filter_by(plan_id=plan.id).order_by(StudyTask.due_date).all()
    
    # Tính toán tiến độ
    total_tasks = len(tasks)
    completed_tasks = sum(1 for task in tasks if task.is_completed)
    progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    return render_template('view_study_plan.html', plan=plan, tasks=tasks, progress=progress)

@app.route('/study-plan/task/<int:task_id>/toggle', methods=['POST'])
@login_required
def toggle_task(task_id):
    task = StudyTask.query.filter_by(id=task_id).first_or_404()
    plan = StudyPlan.query.filter_by(id=task.plan_id, user_id=current_user.id).first_or_404()
    
    # Chuyển đổi trạng thái
    task.is_completed = not task.is_completed
    if task.is_completed:
        task.completed_at = datetime.utcnow()
    else:
        task.completed_at = None
    
    db.session.commit()
    
    # Kiểm tra nếu tất cả task đã hoàn thành
    all_tasks = StudyTask.query.filter_by(plan_id=plan.id).all()
    if all(task.is_completed for task in all_tasks):
        plan.status = 'completed'
        db.session.commit()
        
        # Cập nhật điểm và thông báo
        profile = UserProfile.query.filter_by(user_id=current_user.id).first()
        if profile:
            profile.total_points += 50  # Điểm thưởng khi hoàn thành kế hoạch
            db.session.commit()
            
            # Tạo thông báo
            notification = Notification(
                user_id=current_user.id,
                title="Kế hoạch học tập hoàn thành",
                content=f"Chúc mừng! Bạn đã hoàn thành kế hoạch học tập '{plan.title}'. Bạn nhận được 50 điểm thưởng.",
                type="plan"
            )
            db.session.add(notification)
            db.session.commit()
    
    return redirect(url_for('view_study_plan', plan_id=plan.id))

def generate_auto_study_plan(level):
    """Tạo kế hoạch học tập tự động dựa trên cấp độ"""
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=30)  # Kế hoạch 30 ngày
    
    plan = {
        'title': f'Kế hoạch học tiếng Anh {level} - 30 ngày',
        'description': f'Kế hoạch học tập tự động được tạo bởi AI cho cấp độ {level}',
        'goal': f'Nâng cao trình độ tiếng Anh lên cấp độ tiếp theo sau {level}',
        'start_date': today,
        'end_date': end_date,
        'tasks': []
    }
    
    # Xác định các nhiệm vụ học tập dựa trên cấp độ
    if level in ["Beginner (A1)", "Elementary (A2)"]:
        # Tập trung vào từ vựng cơ bản và ngữ pháp đơn giản
        plan['tasks'] = [
            {
                'title': 'Học 10 từ vựng mỗi ngày',
                'description': 'Sử dụng flashcards hoặc ứng dụng học từ vựng',
                'due_date': today + timedelta(days=7),
                'priority': 1
            },
            {
                'title': 'Luyện nghe với đoạn hội thoại đơn giản',
                'description': 'Nghe mỗi ngày 15 phút',
                'due_date': today + timedelta(days=14),
                'priority': 2
            },
            {
                'title': 'Học các cấu trúc câu cơ bản',
                'description': 'Tập trung vào thì hiện tại đơn, hiện tại tiếp diễn',
                'due_date': today + timedelta(days=21),
                'priority': 2
            },
            {
                'title': 'Thực hành giao tiếp hàng ngày',
                'description': 'Luyện nói các chủ đề đơn giản',
                'due_date': today + timedelta(days=28),
                'priority': 3
            }
        ]
    elif level in ["Intermediate (B1)", "Upper Intermediate (B2)"]:
        # Tập trung vào kỹ năng giao tiếp và ngữ pháp phức tạp hơn
        plan['tasks'] = [
            {
                'title': 'Đọc một bài báo tiếng Anh mỗi ngày',
                'description': 'Chọn các bài báo ở mức độ phù hợp',
                'due_date': today + timedelta(days=7),
                'priority': 2
            },
            {
                'title': 'Học các thì phức tạp',
                'description': 'Tập trung vào quá khứ hoàn thành, tương lai hoàn thành',
                'due_date': today + timedelta(days=14),
                'priority': 1
            },
            {
                'title': 'Luyện nghe với podcast hoặc TED talks',
                'description': 'Nghe và tóm tắt nội dung',
                'due_date': today + timedelta(days=21),
                'priority': 2
            },
            {
                'title': 'Viết nhật ký tiếng Anh hàng ngày',
                'description': 'Viết ít nhất 100 từ mỗi ngày',
                'due_date': today + timedelta(days=28),
                'priority': 3
            },
            {
                'title': 'Tham gia một câu lạc bộ tiếng Anh',
                'description': 'Thực hành giao tiếp với người khác',
                'due_date': today + timedelta(days=30),
                'priority': 3
            }
        ]
    else:  # C1, C2
        # Tập trung vào kỹ năng nâng cao và ngôn ngữ học thuật
        plan['tasks'] = [
            {
                'title': 'Đọc sách tiếng Anh',
                'description': 'Đọc một cuốn sách gốc tiếng Anh',
                'due_date': today + timedelta(days=10),
                'priority': 2
            },
            {
                'title': 'Viết bài luận học thuật',
                'description': 'Viết một bài luận 500-1000 từ',
                'due_date': today + timedelta(days=14),
                'priority': 1
            },
            {
                'title': 'Học ngữ pháp nâng cao',
                'description': 'Tập trung vào cấu trúc phức tạp như đảo ngữ, mệnh đề rút gọn',
                'due_date': today + timedelta(days=21),
                'priority': 2
            },
            {
                'title': 'Phân tích phim/video không có phụ đề',
                'description': 'Xem và phân tích phim, chương trình TV tiếng Anh',
                'due_date': today + timedelta(days=25),
                'priority': 3
            },
            {
                'title': 'Thuyết trình tiếng Anh',
                'description': 'Chuẩn bị và thực hiện một bài thuyết trình',
                'due_date': today + timedelta(days=30),
                'priority': 1
            }
        ]
    
    return plan

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
    
    # Thống kê theo cấp độ người dùng
    user_levels = db.session.query(
        UserProfile.language_level, 
        db.func.count(UserProfile.id)
    ).filter(UserProfile.language_level != None).group_by(UserProfile.language_level).all()
    
    # Thống kê hoạt động trong 7 ngày qua
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    recent_activities = {}
    recent_activities['new_users'] = User.query.filter(User.created_at >= week_ago).count()
    recent_activities['test_results'] = TestResult.query.filter(TestResult.completed_at >= week_ago).count()
    recent_activities['exercise_submissions'] = ExerciseSubmission.query.filter(ExerciseSubmission.submitted_at >= week_ago).count()
    
    return render_template('admin/dashboard.html', 
                          total_users=total_users,
                          total_courses=total_courses,
                          total_lessons=total_lessons,
                          total_tests=total_tests,
                          latest_users=latest_users,
                          latest_courses=latest_courses,
                          user_levels=user_levels,
                          recent_activities=recent_activities)

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
            duration_weeks=duration_weeks,
            topic=topic,
            is_auto_generated=False
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
        topic = request.form.get('topic')
        order = request.form.get('order')
        estimated_time = request.form.get('estimated_time')
        focus_skills = request.form.getlist('focus_skills')
        
        # Tạo nội dung bài học từ AI nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            # Sử dụng cache thay vì gọi trực tiếp AI
            try:
                weakness = request.form.get('weakness')
                ai_content = content_cache.get_lesson(course.level, topic, weakness)
                content = json.dumps(ai_content)
                flash('Đã tạo nội dung bài học bằng AI!', 'success')
            except Exception as e:
                content = request.form.get('content', '{}')
                flash(f'Lỗi khi tạo nội dung bằng AI: {str(e)}. Sử dụng nội dung nhập tay.', 'warning')
        else:
            content = request.form.get('content', '{}')
        
        lesson = Lesson(
            course_id=course.id,
            title=title,
            content=content,
            order=order,
            estimated_time=estimated_time,
            topic=topic,
            focus_skills=json.dumps(focus_skills)
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
        lesson.topic = request.form.get('topic')
        lesson.order = request.form.get('order')
        lesson.estimated_time = request.form.get('estimated_time')
        lesson.focus_skills = json.dumps(request.form.getlist('focus_skills'))
        
        # Tạo lại nội dung bài học từ AI nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            course = Course.query.get(lesson.course_id)
            try:
                weakness = request.form.get('weakness')
                # Sử dụng cache thay vì gọi trực tiếp AI
                ai_content = content_cache.get_lesson(course.level, lesson.topic, weakness)
                lesson.content = json.dumps(ai_content)
                flash('Đã cập nhật nội dung bài học bằng AI!', 'success')
            except Exception as e:
                lesson.content = request.form.get('content', lesson.content)
                flash(f'Lỗi khi tạo nội dung bằng AI: {str(e)}. Giữ nguyên nội dung cũ.', 'warning')
        else:
            lesson.content = request.form.get('content', lesson.content)
        
        lesson.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash('Bài học đã được cập nhật thành công!', 'success')
        return redirect(url_for('admin_course_lessons', course_id=lesson.course_id))
    
    # Parse focus_skills từ JSON
    try:
        focus_skills = json.loads(lesson.focus_skills) if lesson.focus_skills else []
    except:
        focus_skills = []
    
    return render_template('admin/edit_lesson.html', lesson=lesson, focus_skills=focus_skills)

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
        skill = request.form.get('skill')
        
        # Tạo nội dung bài tập từ AI
        if request.form.get('generate_content') == 'yes':
            course = Course.query.get(lesson.course_id)
            try:
                # Sử dụng cache thay vì gọi trực tiếp AI
                ai_content = content_cache.get_topic_test(course.level, lesson.topic, int(request.form.get('num_questions', 10)))
                content = json.dumps(ai_content["questions"])
                flash('Đã tạo nội dung bài tập bằng AI!', 'success')
            except Exception as e:
                content = request.form.get('content', '[]')
                flash(f'Lỗi khi tạo nội dung bằng AI: {str(e)}. Sử dụng nội dung nhập tay.', 'warning')
        else:
            content = request.form.get('content', '[]')
        
        exercise = Exercise(
            lesson_id=lesson.id,
            title=title,
            description=description,
            type=exercise_type,
            content=content,
            time_limit=time_limit,
            points=points,
            skill=skill
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
        exercise.skill = request.form.get('skill')
        
        # Tạo lại nội dung bài tập từ AI
        if request.form.get('generate_content') == 'yes':
            lesson = Lesson.query.get(exercise.lesson_id)
            course = Course.query.get(lesson.course_id)
            try:
                # Sử dụng cache thay vì gọi trực tiếp AI
                ai_content = content_cache.get_topic_test(course.level, lesson.topic, int(request.form.get('num_questions', 10)))
                exercise.content = json.dumps(ai_content["questions"])
                flash('Đã cập nhật nội dung bài tập bằng AI!', 'success')
            except Exception as e:
                exercise.content = request.form.get('content', exercise.content)
                flash(f'Lỗi khi tạo nội dung bằng AI: {str(e)}. Giữ nguyên nội dung cũ.', 'warning')
        else:
            exercise.content = request.form.get('content', exercise.content)
        
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
        instructions = request.form.get('instructions', 'Hãy đọc kỹ và trả lời các câu hỏi.')
        is_active = True if request.form.get('is_active') else False
        
        # Tạo nội dung bài test từ cache
        if test_type == 'placement':
            ai_content = content_cache.get_placement_test()
        else:
            ai_content = content_cache.get_level_test(level)
        
        test = Test(
            title=title,
            description=description,
            type=test_type,
            level=level,
            questions=json.dumps(ai_content["questions"]),
            time_limit=time_limit,
            passing_score=passing_score,
            instructions=instructions,
            is_active=is_active
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
        test.is_active = True if request.form.get('is_active') else False
        test.instructions = request.form.get('instructions')
        
        if test.type != 'placement':
            test.level = request.form.get('level')
        
        test.time_limit = request.form.get('time_limit')
        test.passing_score = request.form.get('passing_score')
        
        # Tạo lại nội dung bài test từ cache nếu yêu cầu
        if request.form.get('generate_content') == 'yes':
            if test.type == 'placement':
                ai_content = content_cache.get_placement_test()
            else:
                ai_content = content_cache.get_level_test(test.level)
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

@app.route('/admin/test/<int:test_id>/results')
@login_required
@admin_required
def admin_test_results(test_id):
    test = Test.query.get_or_404(test_id)
    
    # Lấy tất cả kết quả bài test
    results = TestResult.query.filter_by(test_id=test.id).order_by(TestResult.completed_at.desc()).all()
    
    # Lấy thông tin người dùng
    users = {r.user_id: User.query.get(r.user_id) for r in results}
    
    return render_template('admin/test_results.html', test=test, results=results, users=users)

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
    
    # Tỷ lệ hoàn thành khóa học
    enrolled_users = CourseEnrollment.query.count()
    completed_courses = CourseEnrollment.query.filter_by(status='completed').count()
    completion_rate = (completed_courses / enrolled_users * 100) if enrolled_users > 0 else 0
    
    # Thống kê bài tập theo loại kỹ năng
    skill_exercises = db.session.query(
        Exercise.skill,
        db.func.count(Exercise.id)
    ).group_by(Exercise.skill).all()
    
    # Thống kê bài tập đã nộp theo ngày
    daily_submissions = []
    for i in range(30):
        date = thirty_days_ago + timedelta(days=i)
        count = ExerciseSubmission.query.filter(
            db.func.date(ExerciseSubmission.submitted_at) == date
        ).count()
        daily_submissions.append((date.strftime('%d/%m'), count))
    
    return render_template('admin/reports.html',
                          new_users_count=new_users_count,
                          total_test_results=total_test_results,
                          total_enrollments=total_enrollments,
                          total_study_time=total_study_time,
                          user_levels=user_levels,
                          completion_rate=completion_rate,
                          skill_exercises=skill_exercises,
                          daily_submissions=daily_submissions)

@app.route('/admin/generate-course/<level>', methods=['POST'])
@login_required
@admin_required
def admin_generate_course(level):
    """Tạo khóa học bằng AI cho một trình độ cụ thể"""
    try:
        topic = request.form.get('topic', random.choice(ai.topics))
        
        # Lấy mẫu khóa học từ cache thay vì tạo trực tiếp
        course_template = content_cache.get_course(level)
        
        # Cập nhật topic nếu người dùng chỉ định
        if topic:
            course_template["topic"] = topic
            course_template["title"] = f"Khóa học {topic} - {level}"
        
        # Tạo khóa học mới từ template
        course = Course(
            title=course_template["title"],
            description=course_template["description"],
            level=level,
            is_published=True,
            duration_weeks=8,
            topic=course_template["topic"],
            is_auto_generated=True
        )
        db.session.add(course)
        db.session.commit()
        
        # Tạo các bài học từ template
        for lesson_template in course_template["lessons"]:
            # Cập nhật topic cho lesson nếu cần
            if topic:
                lesson_template["topic"] = topic
                lesson_template["title"] = f"Bài {lesson_template['order']}: {topic} - Phần {lesson_template['order']}"
            
            # Tạo nội dung bài học từ cache
            lesson_content = content_cache.get_lesson(level, lesson_template["topic"])
            
            lesson = Lesson(
                course_id=course.id,
                title=lesson_template["title"],
                content=json.dumps(lesson_content),
                order=lesson_template["order"],
                estimated_time=lesson_template["estimated_time"],
                topic=lesson_template["topic"],
                focus_skills=json.dumps(["listening", "speaking", "reading", "writing"])
            )
            db.session.add(lesson)
            db.session.commit()
            
            # Tạo bài tập tương ứng
            exercise_template = lesson_template.get("exercise", {})
            exercise_questions = content_cache.get_topic_test(level, lesson_template["topic"], 5)
            
            exercise = Exercise(
                lesson_id=lesson.id,
                title=exercise_template.get("title", f"Bài tập: {lesson_template['topic']}"),
                description=exercise_template.get("description", f"Bài tập ôn luyện kiến thức về {lesson_template['topic']}"),
                type=exercise_template.get("type", "quiz"),
                content=json.dumps(exercise_questions["questions"]),
                time_limit=exercise_template.get("time_limit", 15),
                points=exercise_template.get("points", 10)
            )
            db.session.add(exercise)
        
        db.session.commit()
        
        flash(f'Đã tạo thành công khóa học "{course.title}" bằng AI!', 'success')
    except Exception as e:
        flash(f'Lỗi khi tạo khóa học: {str(e)}', 'danger')
    
    return redirect(url_for('admin_courses'))

@app.route('/admin/user-progress')
@login_required
@admin_required
def admin_user_progress():
    """Xem tiến độ học tập của tất cả người dùng"""
    users = User.query.filter_by(role='student').all()
    
    # Thu thập thông tin học tập của mỗi người dùng
    user_progresses = []
    
    for user in users:
        profile = UserProfile.query.filter_by(user_id=user.id).first()
        
        # Số khóa học đã đăng ký
        enrollments = CourseEnrollment.query.filter_by(user_id=user.id).count()
        
        # Số bài tập đã hoàn thành
        completed_exercises = ExerciseSubmission.query.filter_by(user_id=user.id).count()
        
        # Số bài kiểm tra đã làm
        completed_tests = TestResult.query.filter_by(user_id=user.id).count()
        
        # Điểm trung bình
        avg_score = db.session.query(db.func.avg(ExerciseSubmission.score)).filter_by(user_id=user.id).scalar() or 0
        
        user_progresses.append({
            'user': user,
            'profile': profile,
            'enrollments': enrollments,
            'completed_exercises': completed_exercises,
            'completed_tests': completed_tests,
            'avg_score': round(avg_score, 2)
        })
    
    return render_template('admin/user_progress.html', user_progresses=user_progresses)

# Route để quản lý cache từ giao diện admin
@app.route('/admin/content-cache')
@login_required
@admin_required
def admin_content_cache():
    """Quản lý và xem thông tin về cache nội dung"""
    # Thống kê số lượng nội dung trong cache
    cache_stats = {
        'placement_tests': len(content_cache.test_cache['placement']),
        'level_tests': {level: len(tests) for level, tests in content_cache.test_cache['level'].items()},
        'topic_tests': len(content_cache.test_cache.get('topics', {})),
        'lessons': {level: len(topics) for level, topics in content_cache.lesson_cache.items() if topics},
        'courses': {level: len(templates) for level, templates in content_cache.course_cache.items() if templates}
    }
    
    # Thống kê media cache
    media_stats = {
        'audio_files': len(content_cache.media_cache.get('audio', {})),
        'images': len(content_cache.media_cache.get('images', {}))
    }
    
    return render_template('admin/content_cache.html', 
                           cache_stats=cache_stats,
                           media_stats=media_stats,
                           ai_creativity_level=ai.creativity_level)

@app.route('/admin/content-cache/generate', methods=['POST'])
@login_required
@admin_required
def admin_generate_cache():
    """Khởi động quá trình tạo cache mới"""
    cache_type = request.form.get('cache_type')
    level = request.form.get('level')
    count = int(request.form.get('count', 1))
    
    if cache_type == 'placement_test':
        # Tạo bài test xếp loại
        content_cache._generate_placement_tests(count)
        flash(f'Đã tạo {count} bài test xếp loại mới.', 'success')
    
    elif cache_type == 'level_test':
        # Tạo bài test theo cấp độ
        if level in ai.levels:
            content_cache._generate_level_tests(level, count)
            flash(f'Đã tạo {count} bài test cấp độ {level} mới.', 'success')
        else:
            flash('Cấp độ không hợp lệ!', 'danger')
    
    elif cache_type == 'lesson':
        # Tạo bài học theo cấp độ và chủ đề
        if level in ai.levels:
            topic = request.form.get('topic')
            if topic:
                content_cache._generate_lesson(level, topic)
                flash(f'Đã tạo bài học mới cho chủ đề {topic} cấp độ {level}.', 'success')
            else:
                flash('Vui lòng chọn chủ đề!', 'danger')
        else:
            flash('Cấp độ không hợp lệ!', 'danger')
    
    elif cache_type == 'course':
        # Tạo mẫu khóa học
        if level in ai.levels:
            content_cache._generate_course_templates(level, count)
            flash(f'Đã tạo {count} mẫu khóa học mới cho cấp độ {level}.', 'success')
        else:
            flash('Cấp độ không hợp lệ!', 'danger')
    
    elif cache_type == 'media':
        # Tạo media mới
        try:
            content_cache._generate_sample_media()
            flash('Đã tạo các file media mẫu mới.', 'success')
        except Exception as e:
            flash(f'Lỗi khi tạo media: {str(e)}', 'danger')
    
    elif cache_type == 'all':
        # Khởi tạo cache toàn bộ
        thread = content_cache.initialize_cache(background=True)
        flash('Đã bắt đầu tạo cache nội dung đầy đủ. Quá trình này sẽ chạy trong nền.', 'success')
    
    elif cache_type == 'update_creativity':
        # Cập nhật mức độ sáng tạo của AI
        try:
            new_level = float(request.form.get('creativity_level', 0.7))
            if 0.1 <= new_level <= 0.95:
                ai.creativity_level = new_level
                flash(f'Đã cập nhật mức độ sáng tạo AI thành {new_level:.2f}', 'success')
            else:
                flash('Mức độ sáng tạo phải nằm trong khoảng 0.1 - 0.95', 'danger')
        except ValueError:
            flash('Giá trị không hợp lệ!', 'danger')
    
    return redirect(url_for('admin_content_cache'))

@app.route('/admin/content-cache/clear', methods=['POST'])
@login_required
@admin_required
def admin_clear_cache():
    """Xóa cache nội dung"""
    cache_type = request.form.get('cache_type')
    
    if cache_type == 'all':
        # Xóa toàn bộ cache
        content_cache.test_cache = {
            'placement': [],
            'level': {level: [] for level in ai.levels},
            'topics': {}
        }
        content_cache.lesson_cache = {level: {} for level in ai.levels}
        content_cache.course_cache = {level: [] for level in ai.levels}
        content_cache.media_cache = {'audio': {}, 'images': {}}
        content_cache.save_cache()
        flash('Đã xóa toàn bộ cache nội dung.', 'success')
    
    elif cache_type == 'placement_test':
        # Xóa cache bài test xếp loại
        content_cache.test_cache['placement'] = []
        content_cache.save_cache()
        flash('Đã xóa cache bài test xếp loại.', 'success')
    
    elif cache_type == 'level_test':
        # Xóa cache bài test theo cấp độ
        level = request.form.get('level')
        if level in content_cache.test_cache['level']:
            content_cache.test_cache['level'][level] = []
            content_cache.save_cache()
            flash(f'Đã xóa cache bài test cấp độ {level}.', 'success')
        else:
            flash('Cấp độ không hợp lệ!', 'danger')
    
    elif cache_type == 'media':
        # Xóa cache media
        content_cache.media_cache = {'audio': {}, 'images': {}}
        content_cache.save_cache()
        flash('Đã xóa cache media.', 'success')
    
    return redirect(url_for('admin_content_cache'))

@app.route('/admin/discussions')
@login_required
@admin_required
def admin_discussions():
    """Quản lý diễn đàn thảo luận"""
    discussions = Discussion.query.order_by(Discussion.created_at.desc()).all()
    return render_template('admin/discussions.html', discussions=discussions)

@app.route('/admin/discussion/<int:discussion_id>')
@login_required
@admin_required
def admin_view_discussion(discussion_id):
    """Xem chi tiết thảo luận"""
    discussion = Discussion.query.get_or_404(discussion_id)
    comments = DiscussionComment.query.filter_by(discussion_id=discussion.id).order_by(DiscussionComment.created_at).all()
    return render_template('admin/view_discussion.html', discussion=discussion, comments=comments)

@app.route('/admin/discussion/<int:discussion_id>/pin', methods=['POST'])
@login_required
@admin_required
def admin_pin_discussion(discussion_id):
    """Ghim/Bỏ ghim thảo luận"""
    discussion = Discussion.query.get_or_404(discussion_id)
    discussion.is_pinned = not discussion.is_pinned
    db.session.commit()
    
    action = "ghim" if discussion.is_pinned else "bỏ ghim"
    flash(f'Đã {action} thảo luận thành công!', 'success')
    return redirect(url_for('admin_discussions'))

@app.route('/admin/discussion/<int:discussion_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_discussion(discussion_id):
    """Xóa thảo luận"""
    discussion = Discussion.query.get_or_404(discussion_id)
    db.session.delete(discussion)
    db.session.commit()
    
    flash('Đã xóa thảo luận thành công!', 'success')
    return redirect(url_for('admin_discussions'))

@app.route('/admin/feedbacks')
@login_required
@admin_required
def admin_feedbacks():
    """Xem các phản hồi từ người dùng"""
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    return render_template('admin/feedbacks.html', feedbacks=feedbacks)

# API routes
@app.route('/api/audio/<filename>')
def get_audio(filename):
    """API để phát file audio"""
    return send_file(f'static/audio/{filename}')

@app.route('/api/update-study-time', methods=['POST'])
@login_required
def api_update_study_time():
    """API để cập nhật thời gian học tập"""
    if request.method == 'POST':
        minutes = int(request.form.get('minutes', 0))
        if minutes > 0:
            update_study_time(current_user.id, minutes)
            return jsonify({'success': True, 'message': f'Đã cập nhật {minutes} phút học tập'})
    return jsonify({'success': False, 'message': 'Yêu cầu không hợp lệ'})

@app.route('/api/generate-audio', methods=['POST'])
@login_required
def api_generate_audio():
    """API để tạo file audio từ văn bản"""
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text:
            try:
                # Tạo tên file ngẫu nhiên
                filename = f"user_{current_user.id}_{int(time.time())}_{hash(text) % 1000}.mp3"
                
                # Tạo đường dẫn đầy đủ
                audio_dir = os.path.join('static', 'audio', 'user_generated')
                os.makedirs(audio_dir, exist_ok=True)
                audio_path = os.path.join(audio_dir, filename)
                
                # Tạo file audio
                TienIchMedia.tao_audio_tu_van_ban(text, audio_path)
                
                return jsonify({'success': True, 'audio_url': f"/static/audio/user_generated/{filename}"})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})
    return jsonify({'success': False, 'message': 'Yêu cầu không hợp lệ'})

@app.route('/discussions')
@login_required
def discussions():
    """Trang diễn đàn thảo luận"""
    # Lấy các bài thảo luận được ghim
    pinned_discussions = Discussion.query.filter_by(is_pinned=True).order_by(Discussion.updated_at.desc()).all()
    
    # Lấy các bài thảo luận không ghim
    regular_discussions = Discussion.query.filter_by(is_pinned=False).order_by(Discussion.updated_at.desc()).all()
    
    # Lọc theo chủ đề nếu có
    topic_filter = request.args.get('topic')
    if topic_filter:
        regular_discussions = [d for d in regular_discussions if d.topic == topic_filter]
    
    # Danh sách các chủ đề
    topics = db.session.query(Discussion.topic).distinct().all()
    topics = [t[0] for t in topics if t[0]]
    
    return render_template('discussions.html', 
                           pinned_discussions=pinned_discussions, 
                           regular_discussions=regular_discussions,
                           topics=topics,
                           current_topic=topic_filter)

@app.route('/discussion/create', methods=['GET', 'POST'])
@login_required
def create_discussion():
    """Tạo thảo luận mới"""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        topic = request.form.get('topic')
        
        if not title or not content:
            flash('Vui lòng điền đầy đủ tiêu đề và nội dung!', 'danger')
            return redirect(url_for('create_discussion'))
        
        discussion = Discussion(
            title=title,
            content=content,
            topic=topic,
            user_id=current_user.id
        )
        db.session.add(discussion)
        db.session.commit()
        
        flash('Thảo luận đã được tạo thành công!', 'success')
        return redirect(url_for('view_discussion', discussion_id=discussion.id))
    
    return render_template('create_discussion.html')

@app.route('/discussion/<int:discussion_id>')
@login_required
def view_discussion(discussion_id):
    """Xem chi tiết thảo luận"""
    discussion = Discussion.query.get_or_404(discussion_id)
    comments = DiscussionComment.query.filter_by(discussion_id=discussion.id).order_by(DiscussionComment.created_at).all()
    
    return render_template('view_discussion.html', discussion=discussion, comments=comments)

@app.route('/discussion/<int:discussion_id>/comment', methods=['POST'])
@login_required
def add_comment(discussion_id):
    """Thêm bình luận vào thảo luận"""
    discussion = Discussion.query.get_or_404(discussion_id)
    content = request.form.get('content')
    
    if not content:
        flash('Vui lòng nhập nội dung bình luận!', 'danger')
        return redirect(url_for('view_discussion', discussion_id=discussion_id))
    
    comment = DiscussionComment(
        discussion_id=discussion.id,
        user_id=current_user.id,
        content=content
    )
    db.session.add(comment)
    
    # Cập nhật thời gian của thảo luận
    discussion.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    # Tạo thông báo cho chủ thảo luận nếu không phải là người comment
    if discussion.user_id != current_user.id:
        notification = Notification(
            user_id=discussion.user_id,
            title="Có bình luận mới trong thảo luận của bạn",
            content=f"{current_user.username} đã bình luận trong thảo luận '{discussion.title}'",
            type="discussion",
            url=url_for('view_discussion', discussion_id=discussion.id)
        )
        db.session.add(notification)
        db.session.commit()
    
    flash('Bình luận đã được thêm thành công!', 'success')
    return redirect(url_for('view_discussion', discussion_id=discussion_id))

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def user_feedback():
    """Trang gửi phản hồi"""
    if request.method == 'POST':
        content_type = request.form.get('content_type')
        content_id = request.form.get('content_id')
        rating = request.form.get('rating')
        comment = request.form.get('comment')
        
        if not content_type or not rating:
            flash('Vui lòng chọn loại nội dung và đánh giá!', 'danger')
            return redirect(url_for('user_feedback'))
        
        try:
            content_id = int(content_id) if content_id else None
        except ValueError:
            content_id = None
        
        feedback = Feedback(
            user_id=current_user.id,
            content_type=content_type,
            content_id=content_id,
            rating=rating,
            comment=comment
        )
        db.session.add(feedback)
        db.session.commit()
        
        flash('Cảm ơn bạn đã gửi phản hồi!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('feedback.html')

# Utility functions
def update_study_time(user_id, minutes):
    """Cập nhật thời gian học tập cho người dùng"""
    # Cập nhật thống kê ngày hôm nay
    today = datetime.utcnow().date()
    stat = Statistic.query.filter_by(user_id=user_id, date=today).first()
    
    if not stat:
        stat = Statistic(user_id=user_id, date=today)
        db.session.add(stat)
    
    stat.study_time += minutes
    
    # Cập nhật chuỗi ngày học liên tiếp
    update_study_streak(user_id)
    
    db.session.commit()

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

def generate_recommendations(user_id):
    """Tạo gợi ý học tập dựa trên dữ liệu của người dùng"""
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile or not profile.language_level:
        return []
    
    # Lấy điểm yếu từ kết quả test gần nhất
    latest_test = TestResult.query.filter_by(user_id=user_id).order_by(TestResult.completed_at.desc()).first()
    
    weaknesses = []
    if latest_test and latest_test.weaknesses:
        try:
            weaknesses = json.loads(latest_test.weaknesses)
        except:
            pass
    
    # Gợi ý khóa học phù hợp với trình độ
    recommended_courses = Course.query.filter_by(level=profile.language_level, is_published=True).limit(3).all()
    
    # Lấy danh sách khóa học đã đăng ký
    enrolled_courses = [e.course_id for e in CourseEnrollment.query.filter_by(user_id=user_id).all()]
    
    # Lọc bớt các khóa học đã đăng ký
    recommendations = [course for course in recommended_courses if course.id not in enrolled_courses]
    
    # Nếu ít khóa học được gợi ý, thêm khóa học level cao hơn
    if len(recommendations) < 2:
        next_level = None
        if profile.language_level == "Beginner (A1)":
            next_level = "Elementary (A2)"
        elif profile.language_level == "Elementary (A2)":
            next_level = "Intermediate (B1)"
        elif profile.language_level == "Intermediate (B1)":
            next_level = "Upper Intermediate (B2)"
        elif profile.language_level == "Upper Intermediate (B2)":
            next_level = "Advanced (C1)"
        elif profile.language_level == "Advanced (C1)":
            next_level = "Proficient (C2)"
        
        if next_level:
            next_level_courses = Course.query.filter_by(level=next_level, is_published=True).limit(2).all()
            for course in next_level_courses:
                if course.id not in enrolled_courses and course not in recommendations:
                    recommendations.append(course)
    
    return recommendations

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
        fullname='Quản Trị Viên',
        role='admin'
    )
    db.session.add(admin)
    
    # Tạo người dùng học sinh
    student_password = bcrypt.generate_password_hash('student123').decode('utf-8')
    student = User(
        username='student',
        email='student@example.com',
        password=student_password,
        fullname='Học Viên Mẫu',
        role='student'
    )
    db.session.add(student)
    
    db.session.commit()
    
    # Tạo hồ sơ người dùng
    admin_profile = UserProfile(user_id=admin.id)
    student_profile = UserProfile(
        user_id=student.id,
        language_level="Beginner (A1)",
        learning_goals="Tôi muốn học tiếng Anh để du lịch và công việc.",
        preferred_topics="Travel,Work,Culture"
    )
    
    db.session.add_all([admin_profile, student_profile])
    
    # Tạo huy hiệu
    badges = [
        Badge(name="Người mới", description="Hoàn thành bài test xếp loại đầu tiên", requirement_type="tests", requirement_value=1, image="badge_newcomer.png"),
        Badge(name="Siêng năng", description="Học liên tục 7 ngày", requirement_type="streak", requirement_value=7, image="badge_diligent.png"),
        Badge(name="Học giả", description="Đạt 1000 điểm", requirement_type="points", requirement_value=1000, image="badge_scholar.png"),
        Badge(name="Chuyên gia", description="Hoàn thành 5 khóa học", requirement_type="courses", requirement_value=5, image="badge_expert.png")
    ]
    db.session.add_all(badges)
    
    # Tạo khóa học mẫu bằng cache thay vì tạo mới
    beginner_course_template = content_cache.get_course("Beginner (A1)")
    beginner_course = Course(
        title=beginner_course_template["title"],
        description=beginner_course_template["description"],
        level="Beginner (A1)",
        is_published=True,
        duration_weeks=8,
        topic=beginner_course_template["topic"],
        image="default_course.jpg",
        is_auto_generated=True
    )
    
    intermediate_course_template = content_cache.get_course("Intermediate (B1)")
    intermediate_course = Course(
        title=intermediate_course_template["title"],
        description=intermediate_course_template["description"],
        level="Intermediate (B1)",
        is_published=True,
        duration_weeks=10,
        topic=intermediate_course_template["topic"],
        image="default_course.jpg",
        is_auto_generated=True
    )
    
    db.session.add_all([beginner_course, intermediate_course])
    db.session.commit()
    
    # Tạo bài học mẫu cho khóa học beginner
    for i, lesson_template in enumerate(beginner_course_template["lessons"][:2], 1):
        lesson_content = content_cache.get_lesson("Beginner (A1)", lesson_template["topic"])
        
        lesson = Lesson(
            course_id=beginner_course.id,
            title=lesson_template["title"],
            content=json.dumps(lesson_content),
            order=i,
            estimated_time=30,
            topic=lesson_template["topic"],
            focus_skills=json.dumps(["listening", "speaking", "reading", "writing"])
        )
        db.session.add(lesson)
        db.session.commit()
        
        # Tạo bài tập
        exercise_content = content_cache.get_topic_test("Beginner (A1)", lesson_template["topic"], 5)
        exercise = Exercise(
            lesson_id=lesson.id,
            title=f"Bài tập {i}: {lesson_template['topic']}",
            description=f"Bài tập ôn luyện kiến thức về {lesson_template['topic']}",
            type="quiz",
            content=json.dumps(exercise_content["questions"]),
            time_limit=10,
            points=10,
            skill="vocabulary_grammar"
        )
        db.session.add(exercise)
    
    # Tạo bài test mẫu
    placement_test_content = content_cache.get_placement_test()
    placement_test = Test(
        title="Bài test xếp loại",
        description="Bài test này sẽ giúp xác định trình độ tiếng Anh của bạn.",
        type="placement",
        questions=json.dumps(placement_test_content["questions"]),
        time_limit=30,
        passing_score=7.0,
        is_active=True,
        instructions="Hãy làm bài test này để xác định trình độ tiếng Anh của bạn. Bài test bao gồm các phần nghe, nói, đọc, viết và ngữ pháp."
    )
    
    beginner_test_content = content_cache.get_level_test("Beginner (A1)")
    beginner_test = Test(
        title="Kiểm tra trình độ A1",
        description="Bài test đánh giá kiến thức tiếng Anh cơ bản.",
        type="level",
        level="Beginner (A1)",
        questions=json.dumps(beginner_test_content["questions"]),
        time_limit=20,
        passing_score=6.0,
        is_active=True,
        instructions="Bài kiểm tra này đánh giá trình độ A1 của bạn. Hãy hoàn thành tất cả các câu hỏi trong thời gian quy định."
    )
    
    db.session.add_all([placement_test, beginner_test])
    db.session.commit()
    
    # Đăng ký khóa học mẫu cho học sinh
    enrollment = CourseEnrollment(
        user_id=student.id,
        course_id=beginner_course.id,
        last_accessed=datetime.utcnow()
    )
    db.session.add(enrollment)
    
    # Tạo thông báo mẫu
    notification = Notification(
        user_id=student.id,
        title="Chào mừng đến với hệ thống học tiếng Anh AI!",
        content="Chúng tôi rất vui khi bạn tham gia. Hãy bắt đầu với bài test xếp loại để hệ thống AI có thể tạo các khóa học phù hợp với trình độ của bạn.",
        type="system",
        url=url_for('placement_test')
    )
    db.session.add(notification)
    
    # Tạo thảo luận mẫu
    discussion = Discussion(
        title="Làm thế nào để học tiếng Anh hiệu quả?",
        content="Tôi muốn biết các phương pháp học tiếng Anh hiệu quả. Mọi người có thể chia sẻ kinh nghiệm không?",
        user_id=student.id,
        topic="Học tập"
    )
    db.session.add(discussion)
    
    # Tạo bình luận mẫu
    comment = DiscussionComment(
        discussion_id=1,
        user_id=admin.id,
        content="Việc học tiếng Anh đòi hỏi sự kiên trì và thực hành hàng ngày. Bạn nên dành ít nhất 30 phút mỗi ngày để học và thực hành."
    )
    db.session.add(comment)
    
    # Tạo kế hoạch học tập mẫu
    study_plan = StudyPlan(
        user_id=student.id,
        title="Kế hoạch học tiếng Anh 30 ngày",
        description="Kế hoạch học tập 30 ngày để nâng cao trình độ tiếng Anh",
        goal="Nâng cấp độ từ A1 lên A2",
        start_date=datetime.utcnow().date(),
        end_date=datetime.utcnow().date() + timedelta(days=30),
        status='active'
    )
    db.session.add(study_plan)
    db.session.commit()
    
    # Tạo các nhiệm vụ cho kế hoạch học tập
    tasks = [
        StudyTask(
            plan_id=study_plan.id,
            title="Học 10 từ vựng mỗi ngày",
            description="Sử dụng flashcards hoặc ứng dụng học từ vựng",
            due_date=datetime.utcnow().date() + timedelta(days=7),
            priority=1
        ),
        StudyTask(
            plan_id=study_plan.id,
            title="Luyện nghe với đoạn hội thoại đơn giản",
            description="Nghe mỗi ngày 15 phút",
            due_date=datetime.utcnow().date() + timedelta(days=14),
            priority=2
        ),
        StudyTask(
            plan_id=study_plan.id,
            title="Học các cấu trúc câu cơ bản",
            description="Tập trung vào thì hiện tại đơn, hiện tại tiếp diễn",
            due_date=datetime.utcnow().date() + timedelta(days=21),
            priority=2
        )
    ]
    db.session.add_all(tasks)
    
    db.session.commit()
    
    print("Đã tạo dữ liệu mẫu thành công!")

# Tạo cơ sở dữ liệu và khởi chạy ứng dụng
@app.before_first_request
def initialize_db():
    """Tạo cơ sở dữ liệu và dữ liệu mẫu trước request đầu tiên"""
    print("Khởi tạo cơ sở dữ liệu...")
    db.create_all()
    
    # Kiểm tra thư mục mô hình Deepseek
    if os.path.exists('./deepseek-model'):
        print("Đã tìm thấy thư mục mô hình Deepseek")
        model_files = os.listdir('./deepseek-model')
        print(f"Các file trong thư mục mô hình: {', '.join(model_files[:5])}...")
    else:
        print("CẢNH BÁO: Không tìm thấy thư mục mô hình Deepseek")
        print("AI sẽ chạy ở chế độ mô phỏng")
    
    # Tạo ngay ít nhất 1 bài test xếp loại để đảm bảo hoạt động
    print("Tạo dữ liệu cần thiết ban đầu...")
    # Tạo 1 bài test xếp loại
    placement_data = content_cache.get_placement_test()
    # Tạo dữ liệu mẫu
    create_sample_data()
    print("Dữ liệu ban đầu đã được tạo")
    
    # Bắt đầu khởi tạo phần còn lại của cache trong nền
    print("Tiếp tục khởi tạo content cache trong nền...")
    content_cache.initialize_remaining_cache()
    
    # Khởi động scheduler
    try:
        content_cache.start_scheduler()
        print("Content refresh scheduler đã được khởi động")
    except Exception as e:
        print(f"CẢNH BÁO: Không thể khởi động scheduler: {str(e)}")
        print("Ứng dụng vẫn sẽ hoạt động bình thường, nhưng không tự động làm mới nội dung")

# Handler cho các lỗi 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Handler cho các lỗi 500
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Thêm biến toàn cục cho templates
@app.context_processor
def inject_global_vars():
    """Thêm các biến toàn cục cho templates"""
    def get_unread_notifications_count():
        if current_user.is_authenticated:
            return Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
        return 0
    
    return {
        'get_unread_notifications_count': get_unread_notifications_count,
        'current_year': datetime.now().year,
        'app_version': '1.0.0',
        'app_name': 'Học tiếng Anh AI'
    }

# Chạy ứng dụng
if __name__ == '__main__':
    # Tạo thư mục cần thiết
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Khởi tạo AI và cache trước khi chạy ứng dụng
    print("Khởi tạo AI và cache...")
    if not hasattr(ai, 'model_loaded') or not ai.model_loaded:
        print("CẢNH BÁO: Mô hình AI không được tải. Chạy ở chế độ mô phỏng.")
    
    # Khởi động ứng dụng
    app.run(host='0.0.0.0', port=5000, debug=True)