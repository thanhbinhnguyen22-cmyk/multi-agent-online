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
        text = "[Lỗi đọc file PDF " + pdf_path + ": " + str(e) + "]"
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

# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY (GIỐNG WINDOWS EXPLORER) ---
def render_tree_selector(tree, path_prefix="kho_tri_thuc"):
    selected_paths = []
    if not tree:
        return selected_paths

    for name, content in tree.items():
        current_path = os.path.join(path_prefix, name)
        if isinstance(content, dict):
            # Tạo bố cục dòng: Checkbox chọn tất cả nằm bên trái, Expander tên thư mục nằm bên phải
            col_chk, col_exp = st.sidebar.columns([0.15, 0.85])
            with col_chk:
                select_all_folder = st.checkbox("", key="chk_dir_" + current_path, label_visibility="collapsed")
            with col_exp:
                expander = st.sidebar.expander("📁 " + name, expanded=False)
            
            # Chỉ khi mở expander thì nội dung bên trong mới được render gọn gàng bên trong khung
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
            # File PDF nằm ẩn bên trong thư mục, chỉ hiển thị khi người dùng mở thư mục tương ứng
            is_selected = st.checkbox("📄 " + name, key="chk_file_" + content, value=False)
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

if "agents_config_v31" not in st.session_state:
    st.session_state.agents_config_v31 = {
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

active_agents = st.session_state.agents_config_v31["agents"]

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
                st.sidebar.success("Đã thêm agent: " + new_name + "!")
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
    st.sidebar.markdown("**Kết quả tìm kiếm (" + str(len(matched_pdfs)) + "):**")
    if matched_pdfs:
        for p in matched_pdfs:
            if st.sidebar.checkbox("📄 " + os.path.basename(p), key="search_" + p, value=False):
                selected_pdf_files.append(p)
    else:
        st.sidebar.info("Không tìm thấy file phù hợp.")
else:
    knowledge_tree = build_file_tree(actual_kb_dir)
    selected_pdf_files = render_tree_selector(knowledge_tree, path_prefix=actual_kb_dir)

if selected_pdf_files:
    st.sidebar.success("Đã chọn " + str(len(selected_pdf_files)) + " tài liệu tham khảo.")

st.sidebar.divider()
enable_hitl = st.sidebar.checkbox("🛡️ Bật kiểm duyệt thủ công (Human-in-the-Loop)", value=False)

# ================= KHUNG QUẢN LÝ NHẬT KÝ LỖI & TOKEN =================
with st.expander("🛠️ Nhật ký lỗi & Trạng thái Token (" + str(len(st.session_state.error_logs)) + " lỗi ghi nhận)", expanded=False):
    if st.session_state.error_logs:
        st.warning("Hệ thống đã bắt các lỗi token/API bên dưới:")
        for idx, log_item in enumerate(st.session_state.error_logs):
            st.error("**[" + log_item['time'] + "] Agent `" + log_item['agent'] + "`:** " + log_item['error'])
        if st.button("🗑️ Xóa sạch lịch sử lỗi"):
            st.session_state.error_logs = []
            st.rerun()
    else:
        st.info("✨ Chưa có lỗi phát sinh trong phiên làm việc này.")

st.divider()

# ================= GIAO DIỆN CHÍNH (CÓ 3 TAB TRA CỨU) =================
tab1, tab2, tab3 = st.tabs(["🚀 Chạy Hệ Thống", "📜 Lịch sử Phiên làm việc", "📘 Kiến trúc & Tổng kết Hệ thống"])

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
                        with st.spinner("Đang trích xuất văn bản từ " + str(len(selected_pdf_files)) + " tài liệu đã chọn..."):
                            for f_path in selected_pdf_files:
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
                                    "KẾT QUẢ / NGỮ CẢNH TỪ CÁC BƯỚC TRƯỚC:\n"
                                    + current_context + "\n\n"
                                    + "YÊU CẦU GỐC: " + user_query + "\n"
                                    + pdf_section
                                )
                                
                                agent_output = call_groq_with_retry(
                                    client=client,
                                    model=agent_info['model'],
                                    system_prompt=agent_info['system_prompt'],
                                    user_content=prompt_content,
                                    temperature=agent_info.get("temperature", 0.2)
                                )
                        except Exception as e:
                            err_msg = str(e)
                            current_time = datetime.now().strftime("%H:%M:%S")
                            st.session_state.error_logs.append({
                                "time": current_time,
                                "agent": agent_info['name'],
                                "error": err_msg
                            })
                            agent_output = "⚠️ *[Tác nhân gặp sự cố token/API. Xem chi tiết tại Nhật ký lỗi.]*"
                            st.warning("⚠️ " + agent_info['name'] + " đã gặp sự cố. Kiểm tra mục 'Nhật ký lỗi' phía trên.")

                        if enable_hitl:
                            agent_output = st.text_area("Kiểm duyệt/Sửa đổi kết quả của " + agent_info['name'] + ":", value=agent_output, key="edit_" + str(idx))

                        st.write(agent_output)
                        accumulated_context += "\n\n--- Ý kiến từ " + agent_info['name'] + " ---\n" + agent_output
                        final_report += "## Bước " + str(idx+1) + ": " + agent_info['name'] + "\n" + agent_output + "\n\n"
                        time.sleep(0.5)

                    save_to_db("Pipeline (Groq)", user_query, final_report)
                    st.success("🎉 Hoàn tất chuỗi Pipeline và đã lưu lịch sử!")

                    report_filename = "groq_multi_agent_report_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".md"
                    st.download_button(
                        label="📥 Tải xuống Báo cáo Tổng hợp (.md)",
                        data=final_report,
                        file_name=report_filename,
                        mime="text/markdown"
                    )

    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            agent_a_key = st.selectbox("Agent mở đầu", agent_keys, index=0)
        with col_d2:
            agent_b_key = st.selectbox("Agent phản biện", agent_keys, index=min(1, len(agent_keys)-1))
            
        debate_rounds = st.slider("Số vòng tranh luận qua lại", min_value=1, max_value=3, value=2)

        if st.button("⚔️ Bắt đầu Cuộc Tranh Luận", type="primary"):
            if not user_query:
                st.warning("Vui lòng nhập chủ đề tranh luận!")
            elif agent_a_key == agent_b_key:
                st.warning("Vui lòng chọn 2 Agent khác nhau!")
            else:
                api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
                if not api_key:
                    st.error("Chưa cấu hình GROQ_API_KEY trong Secrets!")
                else:
                    client = Groq(api_key=api_key)
                    agent_a = active_agents[agent_a_key]
                    agent_b = active_agents[agent_b_key]

                    debate_history = "# BIÊN BẢN TRANH LUẬN ĐA TÁC NHÂN (GROQ)\n\n**Chủ đề:** " + user_query + "\n\n"

                    for r in range(debate_rounds):
                        round_num = r + 1
                        
                        # --- Agent A ---
                        st.markdown("### 🔵 Vòng " + str(round_num) + ": " + agent_a['name'])
                        try:
                            with st.spinner("Đang phát biểu..."):
                                prompt_a = "LỊCH SỬ TRANH LUẬN:\n" + debate_history + "\n\nNHIỆM VỤ: Lập luận hoặc bảo vệ quan điểm của bạn về chủ đề trên bằng tiếng Việt."
                                text_a = call_groq_with_retry(
                                    client=client,
                                    model=agent_a['model'],
                                    system_prompt=agent_a['system_prompt'],
                                    user_content=prompt_a,
                                    temperature=agent_a.get("temperature", 0.2)
                                )
                        except Exception as e:
                            err_msg = str(e)
                            st.session_state.error_logs.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "agent": agent_a['name'],
                                "error": err_msg
                            })
                            text_a = "⚠️ *[Lỗi xử lý token/API ở vòng này. Xem chi tiết tại Nhật ký lỗi.]*"
                            st.warning("⚠️ " + agent_a['name'] + " gặp lỗi.")

                        debate_history += "## Vòng " + str(round_num) + " - " + agent_a['name'] + "\n" + text_a + "\n\n"
                        st.write(text_a)

                        # --- Agent B ---
                        st.markdown("### 🔴 Vòng " + str(round_num) + ": " + agent_b['name'] + " phản biện")
                        try:
                            with st.spinner("Đang phản biện..."):
                                prompt_b = "LỊCH SỬ TRANH LUẬN:\n" + debate_history + "\n\nNHIỆM VỤ: Chỉ ra điểm yếu, rủi ro hoặc phản bác ý kiến vừa rồi của " + agent_a['name'] + " bằng tiếng Việt."
                                text_b = call_groq_with_retry(
                                    client=client,
                                    model=agent_b['model'],
                                    system_prompt=agent_b['system_prompt'],
                                    user_content=prompt_b,
                                    temperature=agent_b.get("temperature", 0.2)
                                )
                        except Exception as e:
                            err_msg = str(e)
                            st.session_state.error_logs.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "agent": agent_b['name'],
                                "error": err_msg
                            })
                            text_b = "⚠️ *[Lỗi xử lý token/API ở vòng này. Xem chi tiết tại Nhật ký lỗi.]*"
                            st.warning("⚠️ " + agent_b['name'] + " gặp lỗi.")

                        debate_history += "## Vòng " + str(round_num) + " - " + agent_b['name'] + " (Phản biện)\n" + text_b + "\n\n"
                        st.write(text_b)
                        time.sleep(0.5)

                    save_to_db("Debate (Groq)", user_query, debate_history)
                    st.success("🏁 Cuộc tranh luận kết thúc và đã lưu lịch sử!")

                    debate_filename = "groq_debate_report_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".md"
                    st.download_button(
                        label="📥 Tải xuống Biên bản Tranh luận (.md)",
                        data=debate_history,
                        file_name=debate_filename,
                        mime="text/markdown"
                    )

with tab2:
    st.subheader("📜 Lịch sử các phiên làm việc trước đây")
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, mode, query, result FROM history ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            st.info("Chưa có lịch sử phiên làm việc nào.")
        else:
            for row in rows:
                expander_title = "[" + str(row[1]) + "] Chế độ: " + str(row[2]) + " — Yêu cầu: " + str(row[3][:50]) + "..."
                with st.expander(expander_title):
                    st.write("**Yêu cầu gốc:** " + str(row[3]))
                    st.markdown("---")
                    st.markdown(str(row[4]))
    except Exception:
        st.info("Chưa có lịch sử phiên làm việc nào hoặc cơ sở dữ liệu đang khởi tạo.")

# ================= TAB 3: TỔNG KẾT & TRA CỨU KIẾN TRÚC DỰ ÁN =================
with tab3:
    st.header("📘 Bảng Tra cứu Kiến trúc & Thiết kế Hệ thống")
    st.markdown("""
    ---
    ### 🎯 1. Mục tiêu Dự án
    * **Tên ứng dụng:** Multi-Agent Knowledge System (Hệ thống Đa tác nhân phân tích kho tri thức).
    * **Mục đích:** Triển khai ứng dụng web AI kết hợp các Tác nhân chuyên môn (CNTT, ATTT, Pháp chế...) làm việc tuần tự (Pipeline) hoặc tranh luận (Debate) để thẩm định văn bản quy phạm pháp luật/kỹ thuật.
    * **Quy mô dữ liệu:** Kho tài liệu phân cấp (`ATTT`, `CNTT`...), dự kiến **~200 file (.pdf / .docx)** với dung lượng tối đa **~2GB**.

    ---
    ### 🛠️ 2. Công nghệ & Hạ tầng (Tech Stack)
    * **Giao diện:** Streamlit (Layout Wide, đệ quy cây thư mục lồng nhau kiểu Windows Explorer).
    * **AI Inference:** Groq API (Ưu tiên các model hiểu tiếng Việt tốt như `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`).
    * **Xử lý tài liệu:** `pypdf` (đọc PDF) và `python-docx` (đọc Word).
    * **Cơ sở dữ liệu:** `SQLite` (`database.db`) lưu lịch sử và chỉ mục văn bản (chunks).
    * **Quản lý code & Đồng bộ:** GitHub + GitHub Desktop (`PyGithub` cho phép tải file từxa).

    ---
    ### 🏗️ 3. Giải pháp Kiến trúc đã chốt
    * **Cây thư mục lồng nhau:** 
      * Ô checkbox chọn tất cả nằm giữa mũi tên mở rộng và icon thư mục (`[✓] 📁 ATTT`).
      * Thu gọn/mở rộng chuẩn theo cơ chế Windows Explorer (chỉ hiện file con khi mở thư mục).
    * **Tối ưu RAM & GitHub Limit:** 
      * Không đẩy toàn bộ 2GB file thô lên GitHub (tránh nén mạng và tràn RAM Streamlit ~1GB).
      * Thực hiện trích xuất text, cắt chỉ mục (Chunking 500-1000 từ) và lưu vào `SQLite`. Khi chạy online chỉ cần nạp tệp `database.db` đã đóng gói gọn nhẹ.
    * **Đồng bộ Hybrid (Offline & Online):**
      * **Tại PC:** Chạy `streamlit run main.py` cục bộ để đạt hiệu năng tối đa trên ổ cứng 2GB.
      * **Tại Web (Online):** Upload tài liệu trực tiếp qua Streamlit $\rightarrow$ Tự động Push lên GitHub via API $\rightarrow$ Khi bật PC dùng **GitHub Desktop** để kéo (`git pull`) dữ liệu về ổ cứng.
    ---
    """)