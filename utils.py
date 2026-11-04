import os
import time
import random
import streamlit as st
from groq import Groq
import pypdf

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        text = "[Lỗi đọc file PDF " + str(pdf_path) + ": " + str(e) + "]"
    return text

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
                expander = st.expander("📁 " + str(name), expanded=False)
            
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
            is_selected = st.checkbox("📄 " + str(name), key="chk_file_" + str(content), value=False)
            if is_selected:
                selected_paths.append(content)
                
    return list(set(selected_paths))

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