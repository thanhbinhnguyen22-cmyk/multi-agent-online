import os
import sqlite3
import glob
import time
import random
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
        text = f"[Lỗi đọc file PDF: {e}]"
    return text

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
            # Nếu gặp lỗi liên quan đến giới hạn tốc độ hoặc token (429, rate_limit, tokens)
            if ("429" in error_str or "503" in error_str or "rate_limit" in error_str.lower() or "token" in error_str.lower()) and attempt < max_retries - 1:
                sleep_time = (initial_delay * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                raise e

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="Pro Multi-Agent System (Groq)", page_icon="⚡", layout="wide")
st.title("⚡ Hệ thống Đa Tác Nhân Siêu Tốc (Sử dụng Groq API)")
st.write("Hệ thống phối hợp AI siêu tốc tích hợp hạ tầng Groq LPU, quản lý lỗi tập trung và kho tri thức động.")

# Khởi tạo bộ nhớ lưu nhật ký lỗi riêng biệt trong Session State
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

api_key_check = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
AVAILABLE_MODELS = get_available_groq_models(api_key_check) if api_key_check else ["qwen/qwen3.8-27b"]

default_model = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "qwen/qwen3.8-27b"

if "agents_config_v8" not in st.session_state:
    st.session_state.agents_config_v8 = {
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

active_agents = st.session_state.agents_config_v8["agents"]

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
st.sidebar.header("📚 Kho Tri thức Động (PDF)")
pdf_files = glob.glob("knowledge_base/**/*.pdf", recursive=True)
selected_pdf = st.sidebar.selectbox("Chọn tài liệu tham khảo cho phiên này:", ["Không dùng tài liệu"] + pdf_files)

st.sidebar.divider()
enable_hitl = st.sidebar.checkbox("🛡️ Bật kiểm duyệt thủ công (Human-in-the-Loop)", value=False)

# ================= KHUNG QUẢN LÝ NHẬT KÝ LỖI & TOKEN (GỌN GÀNG) =================
with st.expander(f"🛠️ Nhật ký lỗi & Trạng thái Token ({len(st.session_state.error_logs)} lỗi ghi nhận)", expanded=False):
    if st.session_state.error_logs:
        st.warning("Hệ thống đã tự động bắt các lỗi token/API bên dưới để tránh làm gián đoạn màn hình chính:")
        for idx, log_item in enumerate(st.session_state.error_logs):
            st.error(f"**[{log_item['time']}] Agent `{log_item['agent']}`:** {log_item['error']}")
        if st.button("🗑️ Xóa sạch lịch sử lỗi"):
            st.session_state.error_logs = []
            st.rerun()
    else:
        st.info("✨ Chưa có lỗi token hoặc API nào phát sinh trong phiên làm việc này.")

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
                    if selected_pdf != "Không dùng tài liệu" and os.path.exists(selected_pdf):
                        with st.spinner(f"Đang trích xuất văn bản từ {selected_pdf}..."):
                            pdf_content = extract_text_from_pdf(selected_pdf)

                    accumulated_context = ""
                    final_report = f"# BÁO CÁO PHỐI HỢP MULTI-AGENT (GROQ)\n\n**Yêu cầu:** {user_query}\n\n"

                    for idx, agent_key in enumerate(selected_pipeline):
                        agent_info = active_agents[agent_key]
                        st.subheader(f"Bước {idx+1}: {agent_info['name']}")
                        
                        try:
                            with st.spinner(f"Đang phân tích..."):
                                pdf_section = f"\nTÀI LIỆU THAM KHẢO:\n{pdf_content}\n" if (pdf_content and idx == len(selected_pipeline)-1) else ""
                                
                                prompt_content = f"""
                                KẾT QUẢ / NGỮ CẢNH TỪ CÁC BƯỚC TRƯỚC:
                                {accumulated_context if accumulated_context else "Bạn là người mở đầu chuỗi xử lý."}
                                
                                YÊU CẦU GỐC: {user_query}
                                {pdf_section}
                                """
                                
                                agent_output = call_groq_with_retry(
                                    client=client,
                                    model=agent_info['model'],
                                    system_prompt=agent_info['system_prompt'],
                                    user_content=prompt_content,
                                    temperature=agent_info.get("temperature", 0.2)
                                )
                        except Exception as e:
                            # Bắt lỗi token/API gọn gàng, lưu vào log thay vì làm đỏ lòm màn hình
                            err_msg = str(e)
                            current_time = datetime.now().strftime("%H:%M:%S")
                            st.session_state.error_logs.append({
                                "time": current_time,
                                "agent": agent_info['name'],
                                "error": err_msg
                            })
                            agent_output = f"⚠️ *[Tác nhân tạm thời gặp sự cố xử lý token/API. Chi tiết đã được ghi nhận vào mục **Nhật ký lỗi & Trạng thái Token** ở phía trên.]*"
                            st.warning(f"⚠️ {agent_info['name']} đã gặp sự cố giới hạn token hoặc phản hồi. Kiểm tra mục 'Nhật ký lỗi' để xem chi tiết.")

                        if enable_hitl:
                            agent_output = st.text_area(f"Kiểm duyệt/Sửa đổi kết quả của {agent_info['name']}:", value=agent_output, key=f"edit_{idx}")

                        st.write(agent_output)
                        accumulated_context += f"\n\n--- Ý kiến từ {agent_info['name']} ---\n{agent_output}"
                        final_report += f"## Bước {idx+1}: {agent_info['name']}\n{agent_output}\n\n"
                        time.sleep(0.5)

                    save_to_db("Pipeline (Groq)", user_query, final_report)
                    st.success("🎉 Hoàn tất chuỗi Pipeline và đã lưu lịch sử vào cơ sở dữ liệu!")

                    st.download_button(
                        label="📥 Tải xuống Báo cáo Tổng hợp (.md)",
                        data=final_report,
                        file_name=f"groq_multi_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
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

                    debate_history = f"# BIÊN BẢN TRANH LUẬN ĐA TÁC NHÂN (GROQ)\n\n**Chủ đề:** {user_query}\n\n"

                    for r in range(debate_rounds):
                        round_num = r + 1
                        
                        # --- Agent A ---
                        st.markdown(f"### 🔵 Vòng {round_num}: {agent_a['name']}")
                        try:
                            with st.spinner(f"Đang phát biểu..."):
                                prompt_a = f"LỊCH SỬ TRANH LUẬN:\n{debate_history}\n\nNHIỆM VỤ: Lập luận hoặc bảo vệ quan điểm của bạn về chủ đề trên bằng tiếng Việt."
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
                            st.warning(f"⚠️ {agent_a['name']} gặp lỗi token.")

                        debate_history += f"## Vòng {round_num} - {agent_a['name']}\n{text_a}\n\n"
                        st.write(text_a)

                        # --- Agent B ---
                        st.markdown(f"### 🔴 Vòng {round_num}: {agent_b['name']} phản biện")
                        try:
                            with st.spinner(f"Đang phản biện..."):
                                prompt_b = f"LỊCH SỬ TRANH LUẬN:\n{debate_history}\n\nNHIỆM VỤ: Chỉ ra điểm yếu, rủi ro hoặc phản bác ý kiến vừa rồi của {agent_a['name']} bằng tiếng Việt."
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
                            st.warning(f"⚠️ {agent_b['name']} gặp lỗi token.")

                        debate_history += f"## Vòng {round_num} - {agent_b['name']} (Phản biện)\n{text_b}\n\n"
                        st.write(text_b)
                        time.sleep(0.5)

                    save_to_db("Debate (Groq)", user_query, debate_history)
                    st.success("🏁 Cuộc tranh luận kết thúc và đã lưu lịch sử vào cơ sở dữ liệu!")

                    st.download_button(
                        label="📥 Tải xuống Biên bản Tranh luận (.md)",
                        data=debate_history,
                        file_name=f"groq_debate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

with tab2:
    st.subheader("📜 Lịch sử các phiên làm việc trước đây")
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, mode, query, result FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.info("Chưa có lịch sử phiên làm việc nào.")
    else:
        for row in rows:
            with st.expander(f"[{row[1]}] Chế độ: {row[2]} — Yêu cầu: {row[3][:50]}..."):
                st.write(f"**Yêu cầu gốc:** {row[3]}")
                st.markdown("---")
                st.markdown(row[4])