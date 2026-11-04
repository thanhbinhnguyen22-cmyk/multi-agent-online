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

# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY GIỐNG WINDOWS EXPLORER (DÙNG CHO TAB KHO TRI THỨC) ---
def render_tree_selector(tree, path_prefix="kho_tri_thuc"):
    selected_paths = []
    if not tree:
        return selected_paths

    for name, content in tree.items():
        current_path = os.path.join(path_