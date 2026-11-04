import os
import sqlite3
import time
import random
from datetime import datetime
import streamlit as st
from groq import Groq
import pypdf

# --- CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE (LƯU LỊCH SỬ) ---
def init_db():
    try:
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
    except Exception:
        pass

init_db()

def save_to_db(mode, query, result):
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (timestamp, mode, query, result) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, query, result)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

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
        text = "[Lỗi đọc file PDF " + pdf_path + ": " + str(e) + "]"
    return text

# --- HÀM KHỞI TẠO & TÌM KIẾM THƯ MỤC KHO TRI THỨC ---
def get_knowledge_base_path():
    base_dir = "kho_tri_thuc"
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(os.path.join(base_dir, "CNTT"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "ATTT"), exist_ok=True)
        except Exception:
            pass
    return base_dir

# --- HÀM LẤY TOÀN BỘ ĐƯỜNG DẪN FILE PDF ---
def get_all_pdf_paths(base_dir):
    paths = []
    if not os.path.exists(base_dir):
        return paths
    try:
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    paths.append(os.path.join(root, f))
    except Exception:
        pass
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

# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY GIỐNG WINDOWS EXPLORER ---
def render_tree_selector(tree, path_prefix="kho_tri_thuc"):
    selected_paths = []
    if not tree:
        return selected_paths

    for name, content in tree.items():
        current_path = os.path.join(path_prefix, name)
        if isinstance(content, dict):
            col_chk, col_exp = st.columns([0.05, 0.95])
            with col_chk:
                select_all_folder = st.checkbox("", key="chk_dir_" + current_path, label_visibility="collapsed")
            with col_exp:
                expander = st.expander("📁 " + name, expanded=False)
            
            with expander:
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
            is_selected = st.checkbox("📄 " + name, key="chk_file_" + content, value=False)
            if is_selected:
                selected_paths.append(content)
                
    return list(set(selected_paths))

# --- HÀM LẤY MODEL AN TOÀN (KHÔNG GÂY TREO APP) ---
@st.cache_data(ttl=3600)
def get_available_groq_models(api_key):
    default_list = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
    if not api_key:
        return default_list
    try:
        client = Groq(api_key=api_key)
        models_response = client.models.list()
        excluded_keywords = ["embed", "whisper", "guard", "classifier", "rerank", "audio", "speech"]
        model_ids = [
            m.id for m in models_response.data 
            if not any(keyword in m.id.lower() for keyword in excluded_keywords)
        ]
        if model_ids:
            return model_ids
    except Exception:
        pass
    return default_list

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

# Khởi tạo state cho theme nếu chưa có
if "theme" not in st.session_state:
    st.session_state.theme = "Normal"

# ================= HỆ THỐNG DESIGN SYSTEM & CSS (CHUẨN ECHARTS DASHBOARD) =================
echarts_design_system = """
<style>
    /* Import font chữ hiện đại, mượt mà chuẩn dashboard */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Hiệu ứng mượt mà và bo góc cho các khối thành phần (Cards) */
    div.stMetric, div.stExpander, div[data-testid="stVerticalBlock"] > div.element-container {
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    /* Tùy chỉnh thanh Sidebar mang phong cách ECharts Menu */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Hiệu ứng chuyển động (Hover & Active) cho danh mục Menu Sidebar */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-weight: 500;
        transition: all 0.25s ease-in-out;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(64, 158, 255, 0.12);
        color: #409eff;
        transform: translateX(4px);
    }

    /* Tùy chỉnh nút bấm chính (Primary Buttons) */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
    }
</style>
"""

# Phân định màu sắc theo 3 chế độ (Dark, Light, Normal)
if st.session_state.theme == "Light":
    st.markdown(echarts_design_system + """
        <style>
        .stApp { background-color: #f4f6f9; color: #1e293b; }
        section[data-testid="stSidebar"] { background-color: #ffffff; }
        .stTextInput input, .stTextArea textarea, .stSelectbox select { background-color: #ffffff; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "Dark":
    st.markdown(echarts_design_system + """
        <style>
        .stApp { background-color: #0b0f19; color: #f8fafc; }
        section[data-testid="stSidebar"] { background-color: #111827; }
        .stTextInput input, .stTextArea textarea, .stSelectbox select { background-color: #1f2937; color: #f8fafc; border: 1px solid #374151; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown(echarts_design_system, unsafe_allow_html=True)

# ================= THANH SIDEBAR: 3 ICON CHUYỂN ĐỔI NHANH & ĐIỀU HƯỚNG =================
st.sidebar.markdown("### 🎨 Giao diện Dashboard")
col_t1, col_t2, col_t3 = st.sidebar.columns(3)
with col_t1:
    if st.button("🌙", help="Chế độ Tối (Dark)", use_container_width=True):
        st.session_state.theme = "Dark"
        st.rerun()
with col_t2:
    if st.button("☀️", help="Chế độ Sáng (Light)", use_container_width=True):
        st.session_state.theme = "Light"
        st.rerun()
with col_t3:
    if st.button("💻", help="Chế độ Mặc định (Normal)", use_container_width=True):
        st.session_state.theme = "Normal"
        st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### 🧭 Menu Điều Hướng")
navigation = st.sidebar.radio(
    "Chọn chức năng:",
    [
        "🚀 Chạy Hệ Thống", 
        "🌲 Kho Tri Thức", 
        "⚙️ Quản lý Agents", 
        "📜 Lịch sử Phiên làm việc", 
        "📘 Kiến trúc & Tổng kết Hệ thống"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
enable_hitl = st.sidebar.checkbox("🛡️ Bật kiểm duyệt thủ công (HITL)", value=False)

if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

if "selected_pdf_files" not in st.session_state:
    st.session_state.selected_pdf_files = []

with st.sidebar.expander("🛠️ Nhật ký lỗi & Token (" + str(len(st.session_state.error_logs)) + ")", expanded=False):
    if st.session_state.error_logs:
        st.warning("Hệ thống đã bắt các lỗi sau:")
        for log_item in st.session_state.error_logs:
            st.error(f"**[{log_item['time']}] `{log_item['agent']}`:** {log_item['error']}")
        if st.button("🗑️ Xóa sạch lỗi"):
            st.session_state.error_logs = []
            st.rerun()
    else:
        st.info("✨ Chưa có lỗi phát sinh.")

api_key_check = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
AVAILABLE_MODELS = get_available_groq_models(api_key_check)
default_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen/qwen3.8-27b"

if "agents_config_v36" not in st.session_state:
    st.session_state.agents_config_v36 = {
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

active_agents = st.session_state.agents_config_v36["agents"]

# ================= KHU VỰC HIỂN THỊ CHÍNH DỰA TRÊN MENU SIDEBAR =================

st.title("⚡ Hệ thống Đa Tác Nhân Siêu Tốc (Groq AI)")

if navigation == "🚀 Chạy Hệ Thống":
    st.header("🚀 Trung tâm Điều phối Đa Tác Nhân")
    st.write("Phối hợp các tác nhân AI thực thi Pipeline hoặc Tranh luận dựa trên kho tri thức đã chọn.")
    st.divider()

    if st.session_state.selected_pdf_files:
        st.info("📄 **Tài liệu tham khảo đang chọn (" + str(len(st.session_state.selected_pdf_files)) + "):** " + ", ".join([os.path.basename(f) for f in st.session_state.selected_pdf_files[:5]]) + ("..." if len(st.session_state.selected_pdf_files) > 5 else ""))
    else:
        st.warning("⚠️ Hiện chưa có tài liệu nào được chọn từ mục '🌲 Kho Tri Thức'. Hệ thống sẽ phân tích dựa trên tri thức sẵn có của AI.")

    st.divider()
    mode = st.radio("Chọn phương thức hoạt động:", ["🔄 Chuỗi Pipeline tuần tự", "⚔️ Chế độ Tranh luận Vòng lặp (Debate)"])
    st.divider()
    user_query = st.text_area("Nhập yêu cầu hoặc vấn đề cần xử lý bằng tiếng Việt:", placeholder="Ví dụ: Đánh giá phương án triển khai hệ thống...")

    if mode.startswith("🔄"):
        selected_pipeline = st.multiselect(
            "Thứ tự các Agent tham gia:",
            options=list(active_agents.keys()),
            default=list(active_agents.keys()),
            format_func=lambda x: active_agents[x]["name"]
        )
        
        if st.button("🚀 Chạy Chuỗi Pipeline", type="primary"):
            if not user_query:
                st.warning("Vui lòng nhập nội dung yêu cầu!")
            elif not selected_pipeline:
                st.warning("Vui lòng chọn ít nhất một Agent!")
            else:
                api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
                if not api_key:
                    st.error("Chưa cấu hình GROQ_API_KEY trong Secrets!")
                else:
                    client = Groq(api_key=api_key)
                    
                    pdf_content = ""
                    if st.session_state.selected_pdf_files:
                        with st.spinner("Đang trích xuất văn bản từ " + str(len(st.session_state.selected_pdf_files)) + " tài liệu đã chọn..."):
                            for f_path in st.session_state.selected_pdf_files:
                                pdf_content += "\n\n--- TÀI LIỆU: " + os.path.basename(f_path) + " ---\n"
                                pdf_content += extract_text_from_pdf(f_path)

                    accumulated_context = ""
                    final_report = "# BÁO CÁO PHỐI HỢP MULTI-AGENT (GROQ)\n\n**Yêu cầu:** " + user_query + "\n\n"

                    for idx, agent_key in enumerate(selected_pipeline):
                        agent_info = active_agents[agent_key]
                        st.subheader("Bước " + str(idx+1) + ": " + agent_info['name'])
                        
                        try:
                            with st.spinner("Đang phân tích..."):
                                pdf_section = "\n\nTÀI LIỆU THAM KHẢO:\n" + pdf_content + "\n" if (pdf_content and idx == len(selected_pipeline)-1) else ""
                                current_context = accumulated_context if accumulated_context else "Bạn là người mở đầu chuỗi xử lý."
                                
                                prompt_content = (
                                    "KẾT QU