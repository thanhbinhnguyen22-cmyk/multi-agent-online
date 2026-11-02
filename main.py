import os
import json
import sqlite3
import glob
import time
from datetime import datetime
import streamlit as st
from google import genai

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

# --- HÀM GỌI API VỚI CƠ CHẾ AUTO-RETRY NÂNG CAO (5 LẦN THỬ) ---
def call_gemini_with_retry(client, model, contents, config=None, max_retries=5, initial_delay=3):
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model, contents=contents, config=config)
            else:
                return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                st.warning(f"⚠️ Máy chủ đang quá tải (Model: {model}). Đang tự động thử lại lần {attempt + 1}/{max_retries} sau {sleep_time} giây...")
                time.sleep(sleep_time)
            else:
                raise e

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(page_title="Pro Multi-Agent System", page_icon="🤖", layout="wide")
st.title("🤖 Hệ thống Đa Tác Nhân Chuyên Sắc (Pro Multi-Agent)")
st.write("Hệ thống phối hợp AI thông minh tích hợp Đa mô hình chuẩn Gemini 3.8, Kho tri thức động và Auto-Retry chống quá tải.")

# Danh sách các mô hình khả dụng hiện tại
AVAILABLE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-pro"
]

# Khởi tạo dữ liệu cấu hình Agent trong Session State (Mặc định dùng gemini-3.8-flash chuẩn)
if "agents_config" not in st.session_state:
    st.session_state.agents_config = {
        "agents": {
            "it_expert": {
                "name": "Chuyên gia CNTT",
                "model": "gemini-3.8-flash",
                "temperature": 0.3,
                "system_prompt": "Bạn là chuyên gia hạ tầng CNTT. Hãy phân tích kỹ thuật, phần cứng, mạng và giải pháp công nghệ một cách chuẩn xác."
            },
            "security_expert": {
                "name": "Chuyên gia ATTT",
                "model": "gemini-3.8-flash",
                "temperature": 0.1,
                "system_prompt": "Bạn là chuyên gia An toàn thông tin. Hãy kiểm tra các rủi ro bảo mật, lỗ hổng hệ thống và mức độ tuân thủ tiêu chuẩn kỹ thuật."
            },
            "legal_expert": {
                "name": "Chuyên gia Pháp chế",
                "model": "gemini-3.8-flash",
                "temperature": 0.1,
                "system_prompt": "Bạn là chuyên gia luật công nghệ thông tin và văn bản quy phạm pháp luật hành chính nhà nước. Hãy rà soát tính hợp pháp và quy định liên quan."
            }
        }
    }

# ================= THANH SIDEBAR: QUẢN LÝ & CÀI ĐẶT =================
st.sidebar.header("⚙️ Quản lý Đội ngũ Agents")
agent_keys = list(st.session_state.agents_config["agents"].keys())
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
            if new_key in st.session_state.agents_config["agents"]:
                st.sidebar.error("Mã định danh đã tồn tại!")
            else:
                st.session_state.agents_config["agents"][new_key] = {
                    "name": new_name, "model": new_model, "temperature": new_temp, "system_prompt": new_prompt
                }
                st.sidebar.success(f"Đã thêm agent: {new_name}!")
                st.rerun()
        else:
            st.sidebar.error("Vui lòng điền đủ thông tin!")
else:
    if agent_keys:
        chosen_agent = st.sidebar.selectbox("Chọn Agent để chỉnh sửa", agent_keys)
        current_data = st.session_state.agents_config["agents"][chosen_agent]
        
        edit_name = st.sidebar.text_input("Tên hiển thị", value=current_data["name"])
        
        current_model = current_data.get("model", "gemini-3.8-flash")
        model_index = AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0
        edit_model = st.sidebar.selectbox("Chọn Model", AVAILABLE_MODELS, index=model_index)
        
        edit_temp = st.sidebar.slider("Temperature", 0.0, 1.0, float(current_data.get("temperature", 0.2)))
        edit_prompt = st.sidebar.text_area("System Prompt", value=current_data["system_prompt"])
        
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            if st.button("Cập nhật"):
                st.session_state.agents_config["agents"][chosen_agent] = {
                    "name": edit_name, "model": edit_model, "temperature": edit_temp, "system_prompt": edit_prompt
                }
                st.success("Đã cập nhật!")
                st.rerun()
        with col_s2:
            if st.button("Xóa", type="primary"):
                del st.session_state.agents_config["agents"][chosen_agent]
                st.success("Đã xóa!")
                st.rerun()

st.sidebar.divider()
st.sidebar.header("📚 Kho Tri thức Động (PDF)")
pdf_files = glob.glob("knowledge_base/**/*.pdf", recursive=True)
selected_pdf = st.sidebar.selectbox("Chọn tài liệu tham khảo cho phiên này:", ["Không dùng tài liệu"] + pdf_files)

st.sidebar.divider()
enable_hitl = st.sidebar.checkbox("🛡️ Bật kiểm duyệt thủ công (Human-in-the-Loop)", value=False)

# ================= GIAO DIỆN CHÍNH =================
active_agents = st.session_state.agents_config["agents"]
tab1, tab2 = st.tabs(["🚀 Chạy Hệ Thống", "📜 Lịch sử Phiên làm việc"])

with tab1:
    mode = st.radio("Chọn phương thức hoạt động:", ["🔄 Chuỗi Pipeline tuần tự", "⚔️ Chế độ Tranh luận Vòng lặp (Debate)"])
    st.divider()
    user_query = st.text_area("Nhập yêu cầu hoặc vấn đề cần xử lý:", placeholder="Ví dụ: Đánh giá phương án triển khai hệ thống...")

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
                api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Chưa cấu hình GEMINI_API_KEY!")
                else:
                    client = genai.Client(api_key=api_key)
                    uploaded_file = None
                    if selected_pdf != "Không dùng tài liệu" and os.path.exists(selected_pdf):
                        with st.spinner(f"Đang tải tài liệu {selected_pdf} lên hệ thống..."):
                            uploaded_file = client.files.upload(file=selected_pdf)

                    accumulated_context = ""
                    final_report = f"# BÁO CÁO PHỐI HỢP MULTI-AGENT\n\n**Yêu cầu:** {user_query}\n\n"

                    for idx, agent_key in enumerate(selected_pipeline):
                        agent_info = active_agents[agent_key]
                        with st.spinner(f"Bước {idx+1}: [{agent_info['name']}] đang phân tích..."):
                            prompt = f"""
                            {agent_info['system_prompt']}
                            
                            KẾT QUẢ / NGỮ CẢNH TỪ CÁC BƯỚC TRƯỚC:
                            {accumulated_context if accumulated_context else "Bạn là người mở đầu."}
                            
                            YÊU CẦU GỐC: {user_query}
                            """
                            
                            config_param = genai.types.GenerateContentConfig(
                                temperature=agent_info.get("temperature", 0.2)
                            )
                            
                            contents = [uploaded_file, prompt] if (uploaded_file and idx == len(selected_pipeline)-1) else prompt
                            
                            res = call_gemini_with_retry(client, agent_info['model'], contents, config_param)
                            agent_output = res.text

                        st.subheader(f"Bước {idx+1}: {agent_info['name']}")
                        
                        if enable_hitl:
                            agent_output = st.text_area(f"Kiểm duyệt/Sửa đổi kết quả của {agent_info['name']}:", value=agent_output, key=f"edit_{idx}")

                        st.write(agent_output)
                        accumulated_context += f"\n\n--- Ý kiến từ {agent_info['name']} ---\n{agent_output}"
                        final_report += f"## Bước {idx+1}: {agent_info['name']}\n{agent_output}\n\n"

                    save_to_db("Pipeline", user_query, final_report)
                    st.success("🎉 Hoàn tất chuỗi Pipeline và đã lưu lịch sử!")

                    st.download_button(
                        label="📥 Tải xuống Báo cáo Tổng hợp (.md)",
                        data=final_report,
                        file_name=f"multi_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
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
                api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Chưa cấu hình GEMINI_API_KEY!")
                else:
                    client = genai.Client(api_key=api_key)
                    agent_a = active_agents[agent_a_key]
                    agent_b = active_agents[agent_b_key]

                    debate_history = f"# BIÊN BẢN TRANH LUẬN ĐA TÁC NHÂN\n\n**Chủ đề:** {user_query}\n\n"

                    for r in range(debate_rounds):
                        round_num = r + 1
                        
                        with st.spinner(f"Vòng {round_num}: [{agent_a['name']}] đang phát biểu..."):
                            prompt_a = f"""{agent_a['system_prompt']}\n\nLỊCH SỬ:\n{debate_history}\nNHIỆM VỤ: Lập luận hoặc bảo vệ quan điểm."""
                            cfg_a = genai.types.GenerateContentConfig(temperature=agent_a.get("temperature", 0.2))
                            
                            res_a = call_gemini_with_retry(client, agent_a['model'], prompt_a, cfg_a)
                            text_a = res_a.text
                            
                            debate_history += f"## Vòng {round_num} - {agent_a['name']}\n{text_a}\n\n"
                            st.markdown(f"### 🔵 Vòng {round_num}: {agent_a['name']}")
                            st.write(text_a)

                        with st.spinner(f"Vòng {round_num}: [{agent_b['name']}] đang phản biện..."):
                            prompt_b = f"""{agent_b['system_prompt']}\n\nLỊCH SỬ:\n{debate_history}\nNHIỆM VỤ: Chỉ ra điểm yếu/rủi ro trong ý kiến của {agent_a['name']}."""
                            cfg_b = genai.types.GenerateContentConfig(temperature=agent_b.get("temperature", 0.2))
                            
                            res_b = call_gemini_with_retry(client, agent_b['model'], prompt_b, cfg_b)
                            text_b = res_b.text
                            
                            debate_history += f"## Vòng {round_num} - {agent_b['name']} (Phản biện)\n{text_b}\n\n"
                            st.markdown(f"### 🔴 Vòng {round_num}: {agent_b['name']} phản biện")
                            st.write(text_b)

                    save_to_db("Debate", user_query, debate_history)
                    st.success("🏁 Cuộc tranh luận kết thúc và đã lưu lịch sử!")

                    st.download_button(
                        label="📥 Tải xuống Biên bản Tranh luận (.md)",
                        data=debate_history,
                        file_name=f"debate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
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