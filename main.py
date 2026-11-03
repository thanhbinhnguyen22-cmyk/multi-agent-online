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

# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY (CHỈ HIỆN FILE KHI MỞ THƯ MỤC) ---
def render_tree_selector(tree, path_prefix="kho_tri_thuc"):
    selected_paths = []
    if not tree:
        return selected_paths

    for name, content in tree.items():
        current_path = os.path.join(path_prefix, name)
        if isinstance(content, dict):
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

if "agents_config_v18" not in st.session_state:
    st.session_state.agents_config_v18 = {
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

active_agents = st.session_state.agents_config_v18["agents"]

# ================= THANH SIDEBAR: QUẢN LÝ & CÀI ĐẶT =================
st.sidebar.header("⚙️ Quản lý Đội ngũ Agents")
agent_keys = list(active_agents.keys())
selected_action = st.sidebar.selectbox("Hành động", ["✏️ Chỉnh sửa / Sửa tham số", "➕ Thêm Agent mới"])

if selected_action == "➕ Thêm Agent mới":
    st.sidebar.subheader("Thêm Tác Nhân Mới")
    new_key = st.sidebar.text_input("Mã định danh (Key không dấu):")
    new_name = st.sidebar.text_input("Tên hiển thị:")
    new_model = st.sidebar.selectbox("Chọn Model", AVAILABLE_MODELS)
    new_temp = st.sidebar.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.2)
    new_prompt = st.sidebar.text_area("System Prompt:")
    
    if st.sidebar.button("Lưu Agent Mới"):
        if new_key and new_name and new_prompt:
            if new_key in active_agents:
                st.sidebar.error("Mã định danh đã tồn tại!")
            else:
                active_agents[new_key] = {
                    "name": new_name, "model": new_model, "temperature": new_temp, "system_prompt": new_prompt
                }
                st.sidebar.success(f"Đã thêm agent: {new_name}!")
                st.rerun()
        else:
            st.sidebar.error("Vui lòng điền đủ thông tin!")
else:
    if agent_keys:
        chosen_agent = st.sidebar.selectbox("Chọn Agent để chỉnh sửa", agent_keys)
        current_data = active_agents[chosen_agent]
        
        edit_name = st.sidebar.text_input("Tên hiển thị", value=current_data["name"])
        
        current_model = current_data.get("model", default_model)
        if current_model not in AVAILABLE_MODELS:
            current_model = default_model
        model_index = AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0
        edit_model = st.sidebar.selectbox("Chọn Model", AVAILABLE_MODELS, index=model_index)
        
        edit_temp = st.sidebar.slider("Temperature", 0.0, 1.0, float(current_data.get("temperature", 0.2)))
        edit_prompt = st.sidebar.text_area("System Prompt", value=current_data["system_prompt"])
        
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            if st.button("Cập nhật"):
                active_agents[chosen_agent] = {
                    "name": edit_name, "model": edit_model, "temperature": edit_temp, "system_prompt": edit_prompt
                }
                st.success("Đã cập nhật!")
                st.rerun()
        with col_s2:
            if st.button("Xóa", type="primary"):
                del active_agents[chosen_agent]
                st.success("Đã xóa!")
                st.rerun()

st.sidebar.divider()
st.sidebar.header("🌲 Kho Tri thức Động")
actual_kb_dir = get_knowledge_base_path()

search_keyword = st.sidebar.text_input("🔍 Tìm kiếm tài liệu", placeholder="Nhập tên file cần tìm...")

selected_pdf_files = []
if search_keyword:
    all_pdfs = get_all_pdf_paths(actual_kb_dir)
    matched_pdfs = [p for p in all_pdfs if search_keyword.lower() in os.path.basename(p).lower()]
    st.sidebar.markdown(f"**Kết quả tìm kiếm ({len(matched_pdfs)}):**")
    if matched_pdfs:
        for p in matched_pdfs:
            if st.sidebar.checkbox(f"📄 {os.path.basename(p)}", key=f"search_{p}", value=False):
                selected_pdf_files.append(p)
    else:
        st.sidebar.info("Không tìm thấy file phù hợp.")
else:
    knowledge_tree = build_file_tree(actual_kb_dir)
    selected_pdf_files = render_tree_selector(knowledge_tree, path_prefix=actual_kb_dir)

if selected_pdf_files:
    st.sidebar.success(f"Đã chọn {len(selected_pdf_files)} tài liệu tham khảo.")

st.sidebar.divider()
enable_hitl = st.sidebar.checkbox("🛡️ Bật kiểm duyệt thủ công (Human-in-the-Loop)", value=False)

# ================= KHUNG QUẢN LÝ NHẬT KÝ LỖI & TOKEN =================
with st.expander(f"🛠️ Nhật ký lỗi & Trạng thái Token ({len(st.session_state.error_logs)} lỗi ghi nhận)", expanded=False):
    if st.session_state.error_logs:
        st.warning("Hệ thống đã bắt các lỗi token/API bên dưới:")
        for idx, log_item in enumerate(st.session_state.error_logs):
            st.error(f"**[{log_item['time']}] Agent `{log_item['agent']}`:** {log_item['error']}")
        if st.button("🗑️ Xóa sạch lịch sử lỗi"):
            st.session_state.error_logs = []
            st.rerun()
    else:
        st.info("✨ Chưa có lỗi phát sinh trong phiên làm việc này.")

st.divider()

# ================= GIAO DIỆN CHÍNH =================
tab1, tab2 = st.tabs(["🚀 Chạy Hệ Thống", "📜 Lịch sử Phiên làm việc"])

with tab1:
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
                    if selected_pdf_files:
                        with st.spinner(f"Đang trích xuất văn bản từ {len(selected_pdf_files)} tài liệu đã chọn..."):
                            for f_path in selected_pdf_files:
                                pdf_content += f"\n\n--- TÀI LIỆU: {os.path.basename(f_path)} ---\n"
                                pdf_content += extract_text_from_pdf(f_path)

                    accumulated_context = ""
                    final_report = f"# BÁO CÁO PHỐI HỢP MULTI-AGENT (GROQ)\n\n**Yêu cầu:** {user_query}\n\n"

                    for idx, agent_key in enumerate(selected_pipeline):
                        agent_info = active_agents[agent_key]
                        st.subheader(f"Bước {idx+1}: {agent_info['name']}")
                        
                        try:
                            with st.spinner(f"Đang phân tích..."):
                                pdf_section = f"\n\nTÀI LIỆU THAM KHẢO:\n{pdf_content}\n" if (pdf_content and idx == len(selected_pipeline)-1) else ""
                                
                                default_context_msg = "Bạn là người mở đầu chuỗi xử lý."
                                current_context = accumulated_context if accumulated_context else default_context_msg
                                
                                prompt_content = f"""
                                KẾT QUẢ / NGỮ CẢNH TỪ CÁC BƯỚC TRƯỚC:
                                {current_context}
                                
                                YÊU CẦU GỐC: {user_query}