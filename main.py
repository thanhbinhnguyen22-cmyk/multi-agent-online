# --- HÀM HIỂN THỊ CÂY THƯ MỤC ĐỆ QUY GIỐNG WINDOWS EXPLORER (DÙNG CHO TAB KHO TRI THỨC) ---
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