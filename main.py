import os
import time
from datetime import datetime
import streamlit as st
from groq import Groq

from database import init_db, save_to_db
from utils import (
    extract_text_from_pdf,
    get_knowledge_base_path,
    get_all_pdf_paths,
    build_file_tree,
    render_tree_selector,
    get_available_groq_models,
    call_groq_with_retry
)

init_db()

st.set_page_config(page_title="Pro Multi-Agent System (Groq)", page_icon="⚡", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Normal"

# ================= HỆ THỐNG DESIGN SYSTEM & CSS (CHUẨN ECHARTS DASHBOARD) =================
echarts_design_system = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

div.stMetric, div.stExpander, div[data-testid="stVerticalBlock"] > div.element-container {
    border-radius: 12px;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] {
    padding-top: 1rem;
    border-right: 1px solid rgba(128, 128, 128, 0.2);
}

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
            st.error("**[" + str(log_item['time']) + "] `" + str(log_item['agent']) + "`:** " + str(log_item['error']))
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
        file_list_str = ", ".join([os.path.basename(f) for f in st.session_state.selected_pdf_files[:5]])
        if len(st.session_state.selected_pdf_files) > 5:
            file_list_str += "..."
        st.info("📄 **Tài liệu tham khảo đang chọn (" + str(len(st.session_state.selected_pdf_files)) + "):** " + file_list_str)
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
                    final_report = "# BÁO CÁO PHỐI HỢP MULTI-AGENT (GROQ)\n\n**Yêu cầu:** " + str(user_query) + "\n\n"

                    for idx, agent_key in enumerate(selected_pipeline):
                        agent_info = active_agents[agent_key]
                        st.subheader("Bước " + str(idx+1) + ": " + str(agent_info['name']))
                        
                        try:
                            with st.spinner("Đang phân tích..."):
                                pdf_section = "\n\nTÀI LIỆU THAM KHẢO:\n" + pdf_content + "\n" if (pdf_content and idx == len(selected_pipeline)-1) else ""
                                current_context = accumulated_context if accumulated_context else "Bạn là người mở đầu chuỗi xử lý."
                                
                                prompt_content = "KẾT QUẢ / NGỮ CẢNH TỪ CÁC BƯỚC TRƯỚC:\n" + current_context + "\n\nYÊU CẦU GỐC: " + str(user_query) + "\n" + pdf_section
                                
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
                            st.warning("⚠️ " + str(agent_info['name']) + " đã gặp sự cố. Kiểm tra mục 'Nhật ký lỗi' phía trên.")

                        if enable_hitl:
                            agent_output = st.text_area("Kiểm duyệt/Sửa đổi kết quả của " + str(agent_info['name']) + ":", value=agent_output, key="edit_" + str(idx))

                        st.write(agent_output)
                        accumulated_context += "\n\n--- Ý kiến từ " + str(agent_info['name']) + " ---\n" + str(agent_output)
                        final_report += "## Bước " + str(idx+1) + ": " + str(agent_info['name']) + "\n" + str(agent_output) + "\n\n"
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
            agent_a_key = st.selectbox("Agent mở đầu", list(active_agents.keys()), index=0, format_func=lambda x: active_agents[x]["name"])
        with col_d2:
            agent_b_key = st.selectbox("Agent phản biện", list(active_agents.keys()), index=min(1, len(active_agents)-1), format_func=lambda x: active_agents[x]["name"])
            
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

                    debate_history = "# BIÊN BẢN TRANH LUẬN ĐA TÁC NHÂN (GROQ)\n\n**Chủ đề:** " + str(user_query) + "\n\n"

                    for r in range(debate_rounds):
                        round_num = r + 1
                        
                        st.markdown("### 🔵 Vòng " + str(round_num) + ": " + str(agent_a['name']))
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
                            st.warning("⚠️ " + str(agent_a['name']) + " gặp lỗi.")

                        debate_history += "## Vòng " + str(round_num) + " - " + str(agent_a['name']) + "\n" + str(text_a) + "\n\n"
                        st.write(text_a)

                        st.markdown("### 🔴 Vòng " + str(round_num) + ": " + str(agent_b['name']) + " phản biện")
                        try:
                            with st.spinner("Đang phản biện..."):
                                prompt_b = "LỊCH SỬ TRANH LUẬN:\n" + debate_history + "\n\nNHIỆM VỤ: Chỉ ra điểm yếu, rủi ro hoặc phản bác ý kiến vừa rồi của " + str(agent_a['name']) + " bằng tiếng Việt."
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
                            st.warning("⚠️ " + str(agent_b['name']) + " gặp lỗi.")

                        debate_history += "## Vòng " + str(round_num) + " - " + str(agent_b['name']) + " (Phản biện)\n" + str(text_b) + "\n\n"
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

elif navigation == "🌲 Kho Tri Thức":
    st.header("🌲 Kho Tri Thức Động")
    st.write("Duyệt cây thư mục tài liệu hoặc tìm kiếm nhanh file PDF cần tham khảo cho các tác nhân AI.")
    st.divider()
    
    actual_kb_dir = get_knowledge_base_path()
    search_keyword = st.text_input("🔍 Tìm kiếm nhanh tài liệu:", placeholder="Nhập tên file cần tìm...")

    selected_files_in_tab = []
    if search_keyword:
        all_pdfs = get_all_pdf_paths(actual_kb_dir)
        matched_pdfs = [p for p in all_pdfs if search_keyword.lower() in os.path.basename(p).lower()]
        st.markdown("**Kết quả tìm kiếm (" + str(len(matched_pdfs)) + "):**")
        if matched_pdfs:
            for p in matched_pdfs:
                if st.checkbox("📄 " + os.path.basename(p), key="tab_search_" + str(p), value=False):
                    selected_files_in_tab.append(p)
        else:
            st.info("Không tìm thấy file phù hợp.")
    else:
        knowledge_tree = build_file_tree(actual_kb_dir)
        selected_files_in_tab = render_tree_selector(knowledge_tree, path_prefix=actual_kb_dir)

    st.session_state.selected_pdf_files = selected_files_in_tab
    
    if st.session_state.selected_pdf_files:
        st.success("✅ Đã chọn " + str(len(st.session_state.selected_pdf_files)) + " tài liệu để đưa vào phân tích.")
    else:
        st.info("ℹ️ Chưa chọn tài liệu nào. Bạn có thể chọn toàn bộ thư mục bằng checkbox hoặc chọn từng tệp lẻ.")

elif navigation == "⚙️ Quản lý Agents":
    st.header("⚙️ Quản lý Đội ngũ Agents")
    st.write("Thêm mới, cấu hình tham số, chỉnh sửa system prompt hoặc xóa các tác nhân AI trong hệ thống.")
    st.divider()
    
    agent_keys = list(active_agents.keys())
    selected_action = st.selectbox("Chọn thao tác quản lý", ["✏️ Chỉnh sửa / Cấu hình tham số", "➕ Thêm Agent mới"])

    if selected_action == "➕ Thêm Agent mới":
        st.subheader("Thêm Tác Nhân Mới vào Hệ thống")
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            new_key = st.text_input("Mã định danh (Key không dấu, viết liền):", placeholder="ví dụ: financial_expert")
        with col_n2:
            new_name = st.text_input("Tên hiển thị:", placeholder="ví dụ: Chuyên gia Tài chính")
            
        new_model = st.selectbox("Chọn Model Groq", AVAILABLE_MODELS)
        new_temp = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.2)
        new_prompt = st.text_area("System Prompt (Vai trò & Nhiệm vụ):", placeholder="Bạn là chuyên gia tài chính...")
        
        if st.button("💾 Lưu Agent Mới", type="primary"):
            if new_key and new_name and new_prompt:
                if new_key in active_agents:
                    st.error("Mã định danh này đã tồn tại! Vui lòng chọn key khác.")
                else:
                    active_agents[new_key] = {
                        "name": new_name, "model": new_model, "temperature": new_temp, "system_prompt": new_prompt
                    }
                    st.success("Đã thêm thành công agent: " + str(new_name) + "!")
                    st.rerun()
            else:
                st.error("Vui lòng điền đầy đủ các thông tin bắt buộc!")
    else:
        if agent_keys:
            chosen_agent = st.selectbox("Chọn Agent để cấu hình", agent_keys, format_func=lambda x: active_agents[x]["name"])
            current_data = active_agents[chosen_agent]
            
            edit_name = st.text_input("Tên hiển thị", value=current_data["name"])
            
            current_model = current_data.get("model", default_model)
            if current_model not in AVAILABLE_MODELS:
                current_model = default_model
            model_index = AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0
            edit_model = st.selectbox("Chọn Model", AVAILABLE_MODELS, index=model_index)
            
            edit_temp = st.slider("Temperature", 0.0, 1.0, float(current_data.get("temperature", 0.2)))
            edit_prompt = st.text_area("System Prompt", value=current_data["system_prompt"])
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("💾 Cập nhật Thay đổi", type="primary"):
                    active_agents[chosen_agent] = {
                        "name": edit_name, "model": edit_model, "temperature": edit_temp, "system_prompt": edit_prompt
                    }
                    st.success("Đã cập nhật thông tin agent thành công!")
                    st.rerun()
            with col_b2:
                if st.button("🗑️ Xóa Agent này", type="primary"):
                    if len(active_agents) <= 1:
                        st.error("Hệ thống phải có ít nhất 1 Agent hoạt động!")
                    else:
                        del active_agents[chosen_agent]
                        st.success("Đã xóa agent khỏi hệ thống!")
                        st.rerun()
        else:
            st.info("Hiện không có agent nào trong hệ thống.")

elif navigation == "📜 Lịch sử Phiên làm việc":
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

elif navigation == "📘 Kiến trúc & Tổng kết Hệ thống":
    st.header("📘 Bảng Tra cứu Kiến trúc & Thiết kế Hệ thống")
    st.markdown("""
    ---
    ### 🎯 1. Mục tiêu Dự án
    * **Tên ứng dụng:** Multi-Agent Knowledge System (Hệ thống Đa tác nhân phân tích kho tri thức).
    * **Mục đích:** Triển khai ứng dụng web AI kết hợp các Tác nhân chuyên môn (CNTT, ATTT, Pháp chế...) làm việc tuần tự (Pipeline) hoặc tranh luận (Debate) để thẩm định văn bản quy phạm pháp luật/kỹ thuật.
    * **Quy mô dữ liệu:** Kho tài liệu phân cấp (`ATTT`, `CNTT`...), dự kiến **~200 file (.pdf / .docx)** với dung lượng tối đa **~2GB**.

    ---
    ### 🛠️ 2. Công nghệ & Hạ tầng (Tech Stack)
    * **Giao diện:** Streamlit (Layout Wide, menu sidebar điều hướng với hiệu ứng hover/active hiện đại, bộ 3 icon chuyển đổi Theme nhanh).
    * **AI Inference:** Groq API (Ưu tiên các model hiểu tiếng Việt tốt như `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`).
    * **Xử lý tài liệu:** `pypdf` (đọc PDF) và `python-docx` (đọc Word).
    * **Cơ sở dữ liệu:** `SQLite` (`database.db`) lưu lịch sử và chỉ mục văn bản (chunks).
    * **Quản lý code & Đồng bộ:** GitHub + GitHub Desktop (`PyGithub` cho phép tải file từ xa).

    ---
    ### 🏗️ 3. Giải pháp Kiến trúc đã chốt
    * **Cây thư mục lồng nhau:** 
      * Ô checkbox chọn tất cả nằm trước tên thư mục (`[✓] 📁 ATTT`).
      * Thu gọn/mở rộng chuẩn theo cơ chế Windows Explorer (chỉ hiện file con khi mở thư mục).
    * **Tối ưu RAM & GitHub Limit:** 
      * Không đẩy toàn bộ 2GB file thô lên GitHub (tránh nghẽn mạng và tràn RAM Streamlit ~1GB).
      * Thực hiện trích xuất text, cắt chỉ mục (Chunking 500-1000 từ) và lưu vào `SQLite`. Khi chạy online chỉ cần nạp tệp `database.db` đã đóng gói gọn nhẹ.
    * **Đồng bộ Hybrid (Offline & Online):**
      * **Tại PC:** Chạy `streamlit run main.py` cục bộ để đạt hiệu năng tối đa trên ổ cứng 2GB.
      * **Tại Web (Online):** Upload tài liệu trực tiếp qua Streamlit $\rightarrow$ Tự động Push lên GitHub via API $\rightarrow$ Khi bật PC dùng **GitHub Desktop** để kéo (`git pull`) dữ liệu về ổ cứng.
    ---
    """)