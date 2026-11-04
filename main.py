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

# ================= HỆ THỐNG DESIGN SYSTEM & CSS (CHUẨN ECHARTS DASHBOARD & PILL MENU) =================
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

/* Biến st.sidebar.radio thành dạng Menu Pill hiện đại giống hình mẫu */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 4px;
    font-weight: 500;
    transition: all 0.25s ease-in-out;
    cursor: pointer;
    border: none !important;
}

/* Ẩn hoàn toàn ô tròn (radio circle) mặc định của Streamlit */
section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
    display: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    font-size: 15px;
}

/* Hiệu ứng khi di chuột (Hover) */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(128, 128, 128, 0.15);
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

# Tùy chỉnh màu sắc hiệu ứng sáng lên (Active Pill) theo từng Theme
if st.session_state.theme == "Light":
    st.markdown(echarts_design_system + """
    <style>
    .stApp { background-color: #f4f6f9; color: #1e293b; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    .stTextInput input, .stTextArea textarea, .stSelectbox select { background-color: #ffffff; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 8px; }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "Dark":
    st.markdown(echarts_design_system + """
    <style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    .stTextInput input, .stTextArea textarea, .stSelectbox select { background-color: #1f2937; color: #f8fafc; border: 1px solid #374151; border-radius: 8px; }
    
    /* Mục được chọn sẽ sáng lên với lớp nền pill bo tròn nổi bật giống hình mẫu */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown(echarts_design_system + """
    <style>
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(128, 128, 128, 0.2) !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True