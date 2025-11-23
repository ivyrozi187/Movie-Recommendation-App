import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- KHỞI TẠO BIẾN TRẠẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'

# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU & HELPERS
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """Hàm helper để tải dữ liệu CSV với cache."""
    return pd.read_csv(file_path).fillna("")

def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)
    
@st.cache_resource # Chỉ tải dữ liệu tĩnh một lần
def load_and_preprocess_static_data():
    """Tải và tiền xử lý dữ liệu tĩnh (movies và mô hình)."""
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # 1. Tiền xử lý cho Content-Based (TF-IDF/Cosine Sim)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến
        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        return df_movies, cosine_sim_matrix

    except Exception as e:
        st.error(f"LỖI TẢI HOẶC XỬ LÝ DỮ LIỆU TĨNH: {e}. Vui lòng kiểm tra các file CSV.")
        st.stop()


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu người dùng vào Session State."""
    # CHỈ CHẠY 1 LẦN KHI APP KHỞI ĐỘNG
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            df_users.columns = [col.strip() for col in df_users.columns]
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
        except:
             df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất'])

        st.session_state['df_users'] = df_users
    
    return st.session_state['df_users']

def get_unique_movie_titles(df_movies):
    """Lấy danh sách các tên phim duy nhất."""
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

def register_new_user_form(df_movies):
    """Form đăng ký người dùng mới (Lưu vào bộ nhớ Streamlit)."""
    st.header("📝 Đăng Ký Tài Khoản Mới (Phiên Tạm Thời)")
    st.info("📢 Người dùng mới sẽ chỉ tồn tại trong phiên làm việc hiện tại của bạn.")

    df_users = st.session_state['df_users']
    movie_titles_list = get_unique_movie_titles(df_movies)

    with st.form("register_form"):
        username = st.text_input("Tên người dùng mới (Duy nhất):").strip()

        st.subheader("Chọn Phim Đã Xem (Tối thiểu 5 phim để có hồ sơ tốt)")
        
        recent_list_raw = st.multiselect(
            "🎥 5 Phim Đã Xem Gần Nhất:",
            options=movie_titles_list,
            key='recent_select',
            default=movie_titles_list[:5] if len(movie_titles_list) >= 5 else []
        )
        
        favorite_movie = st.selectbox(
            "⭐ Phim Yêu Thích Nhất:",
            options=movie_titles_list,
            key='favorite_select'
        )
        
        submitted = st.form_submit_button("Đăng Ký & Đăng Nhập")

        if submitted:
            # 1. Kiểm tra đầu vào
            if not username:
                st.error("Vui lòng nhập tên người dùng.")
                return
            
            if username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên người dùng '{username}' đã tồn tại.")
                return
            
            if len(recent_list_raw) < 5:
                 st.warning("Vui lòng chọn tối thiểu 5 phim đã xem gần nhất.")
                 return
            
            # 2. Tạo ID mới
            max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
            new_id = int(max_id) + 1
            
            # 3. Tạo dữ liệu mới
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': [str(recent_list_raw)], 
                'Phim yêu thích nhất': [favorite_movie]
            }
            new_user_df = pd.DataFrame(new_user_data)
            
            # 4. CẬP NHẬT SESSION STATE (KHÔNG GHI FILE)
            st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
            
            # 5. Đăng nhập
            st.session_state['logged_in_user'] = username
            st.success(f"🎉 Đăng ký và đăng nhập thành công! Chào mừng, {username}.")
            st.rerun()

def login_form():
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")
    
    df_users = st.session_state['df_users']

    with st.form("login_form"):
        username = st.text_input("Tên người dùng:").strip()
        submitted = st.form_submit_button("Đăng Nhập")
        
        if submitted:
            # DÙNG .values RẤT QUAN TRỌNG ĐỂ KIỂM TRA TÊN CHÍNH XÁC
            if username in df_users['Tên người dùng'].values: 
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Đăng nhập thành công! Chào mừng, {username}.")
                st.rerun()
            else:
                st.error("❌ Tên người dùng không tồn tại.")

def authentication_page(df_movies):
    """Trang Xác thực (chọn Đăng nhập hoặc Đăng ký)."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Đăng Nhập", key="btn_login"):
            st.session_state
