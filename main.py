import os
import sqlite3
import glob
import time
import random
import unicodedata
from datetime import datetime
import streamlit as st
from groq import Groq
import pypdf

# --- CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE (LƯU LỊCH SỬ) ---
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            query TEXT,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_db(mode, query, result):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (timestamp, mode, query, result) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, query, result)
    )
    conn.commit()
    conn.close()

# --- HÀM ĐỌC NỘI DUNG TỪ FILE PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        text = f"[Lỗi đọc file PDF {pdf_path}: {e}]"
    return text

# --- HÀM TÌM KIẾM THƯ MỤC KHO TRI THỨC ---
def get_knowledge_base_path():
    possible_names = ["kho_tri_thuc", "kho tri thuc", "Kho tri thuc", "knowledge_base", "Kho tri thức"]
    for name in possible_names:
        if os.path.exists(name) and os.path.isdir(name):
            return name
            
    try:
        for entry in os.scandir("."):
            if entry.is_dir() and not entry.name.startswith("."):
                name_lower = entry.name.lower()
                if "kho" in name_lower or "tri" in name_lower or "thuc" in name_lower or "knowledge" in name_lower:
                    return entry.name
    except Exception:
        pass
    return "kho_tri_thuc"

# --- HÀM LẤY TOÀN BỘ ĐƯỜNG DẪN FILE PDF ĐỂ TÌM KIẾM ---
def get_all_pdf_paths(base_dir):
    paths = []
    if not os.path.exists(base_dir):
        return paths
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(root, f))
    return paths

# --- HÀM XÂY DỰNG CÂY THƯ MỤC KHO TRI THỨC ---
def build_file_tree(base_dir):
    tree = {}
    if not os.path.exists(base_dir):
        return tree
    try:
        for entry in os.scandir(base_dir):
            if entry.is_dir() and not entry.name.startswith("."):
                tree[entry.name] = build_file_tree(entry.path)
            elif entry.is_file() and entry.name.lower().endswith(".pdf"):
                tree[entry.name] = entry.path
    except Exception:
        pass
    return tree

# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY (MẶC ĐỊNH THU GỌN) ---
def render_tree_selector(tree, path_prefix="kho_tri_thuc"):
    selected_paths = []
    if not tree:
        return selected_paths

    for name, content in tree.items():
        current_path = os.path.join(path_prefix, name)
        if isinstance(content, dict):
            # expanded=False giúp thư mục đóng mặc định, chỉ hiện file khi bấm mở
            with st.sidebar.expander(f"📁 {name}", expanded=False):
                select_all_folder = st.checkbox(f"📁 Chọn tất cả trong '{name}'", key=f"chk_dir_{current_path}")
                sub_selected = render_tree_selector(content, current_path)
                
                if select_all_folder:
                    def get_all_leaves(d):
                        files = []
                        for k, v in d.items():
                            if isinstance(v, dict):
                                files.extend(get_all_leaves(v))
                            else:
                                files.append(v)
                        return files
                    selected_paths.extend(get_all_leaves(content))
                else:
                    selected_paths.extend(sub_selected)
        else:
            is_selected = st.sidebar.checkbox(f"📄 {name}", key=f"chk_file_{content}", value=False)
            if is_selected:
                selected_paths.append(content)
                
    return list(set(selected_paths))

# --- HÀM LẤY VÀ SẮP XẾP MODEL ƯU TIÊN CHO TIẾNG VIỆT ---
def get_available_groq_models(api_key):
    try:
        client = Groq(api_key=api_key)
        models_response = client.models.list()
        
        excluded_keywords = ["embed", "whisper", "guard", "classifier", "rerank", "audio", "speech"]
        model_ids = [
            m.id for m in models_response.data 
            if not any(keyword in m.id.lower() for keyword in excluded_keywords)
        ]
        
        def model_priority(model_name):
            name = model_name.lower()
            if "qwen" in name:
                return 0
            if "gpt-oss-120b" in name:
                return 1
            if "gpt-oss-20b" in name:
                return 2
            return 5
            
        model_ids.sort(key=model_priority)
        
        if model_ids:
            return model_ids
    except Exception:
        pass
    return ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

# --- HÀM GỌI GROQ API VỚI CƠ CHẾ AUTO-RETRY + JITTER ---
def call_groq_with_retry(client, model, system_prompt, user_content, temperature=0.2, max_retries=5, initial_delay=2):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "503" in error_str or "rate_limit" in error_str.lower() or "token" in error_str.lower()) and attempt < max_retries - 1:
                sleep_time = (initial_delay * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                raise e

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="Pro Multi-Agent System (Groq)", page_icon="⚡", layout="wide")
st.title("⚡ Hệ thống Đa Tác Nhân Siêu Tốc (Sử dụng Groq API)")
st.write("Hệ thống phối hợp AI đa tác nhân tích hợp hạ tầng Groq LPU, Cây thư mục tri thức động và Quản lý lỗi tập trung.")

if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

api_key_check = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
AVAILABLE_MODELS = get_available_groq_models(api_key_check) if api_key_check else ["qwen/qwen3.8-27b"]

default_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen/qwen3.8-27b"

if "agents_config_v15" not in st.session_state:
    st.session_state.agents_config_v15 = {
        "agents": {
            "it_expert": {
                "name": "Chuyên gia CNTT",
                "model": default_model,
                "temperature": 0.3,
                "system_prompt": "Bạn là chuyên gia hạ tầng CNTT. Hãy phân tích kỹ thuật, phần cứng, mạng và giải pháp công nghệ một cách chuẩn xác bằng tiếng Việt."
            },
            "security_expert": {
                "name": "Chuyên gia ATTT",
                "model": default_model,
                "temperature": 0.1,
                "system_prompt": "Bạn là chuyên gia An toàn thông tin. Hãy kiểm tra các rủi ro bảo mật, lỗ hổng hệ thống và mức độ tuân thủ tiêu chuẩn kỹ thuật bằng tiếng Việt."
            },
            "legal_expert": {
                "name": "Chuyên gia Pháp chế",
                "model": default_model,
                "temperature": 0.1,
                "system_prompt": "Bạn là chuyên gia luật công nghệ thông tin và văn bản quy phạm pháp luật hành chính nhà nước. Hãy rà soát tính hợp pháp và quy định liên quan bằng tiếng Việt."
            }
        }
    }

active_agents = st.session_state.agents_config_v15["agents"]

# ================= THANH SIDEBAR: QUẢN LÝ & CÀI ĐẶT =================