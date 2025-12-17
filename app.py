import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys
import random
import matplotlib.colors as mcolors
import os # Dùng để kiểm tra và lưu file CSV
from datetime import datetime
from collections import Counter # Thêm từ code mới của bạn

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "data_phim_full_images.csv" 

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" 

# --- CẤU HÌNH DANH SÁCH THỂ LOẠI (TOPICS) THEO YÊU CẦU ---
# Danh sách màu sắc cho Light Theme (Vibrant, Bắt mắt)
COLOR_PALETTE = [
    ("#00BCD4", "#26C6DA", "#00AABF"), # Cyan/Teal (Primary Theme)
    ("#FF5722", "#FF7043", "#E64A19"), # Deep Orange
    ("#4CAF50", "#81C784", "#388E3C"), # Green
    ("#9C27B0", "#BA68C8", "#7B1FA2"), # Purple
    ("#FFC107", "#FFD54F", "#FFB300"), # Amber
    ("#2196F3", "#64B5F6", "#1976D2"), # Blue
    ("#E91E63", "#F06292", "#C2185B"), # Pink
    ("#8BC34A", "#AED581", "#689F38"), # Light Green
    ("#009688", "#4DB6AC", "#00796B"), # Teal
    ("#FF9800", "#FFB74D", "#FB8C00"), # Orange (Accent)
    ("#795548", "#A1887F", "#5D4037"), # Brown
    ("#607D8B", "#90A4AE", "#455A64"), # Blue Grey
]

# Danh sách 23 thể loại từ dữ liệu
GENRES_VI = [
    "Phim Hành Động", "Phim Giả Tượng", "Phim Hài", "Phim Kinh Dị", 
    "Phim Phiêu Lưu", "Phim Chính Kịch", "Phim Khoa Học Viễn Tưởng", 
    "Phim Gây Cấn", "Phim Gia Đình", "Phim Hoạt Hình", "Phim Lãng Mạn", 
    "Phim Tài Liệu", "Phim Chiến Tranh", "Phim Bí Ẩn", "Phim Hình Sự", 
    "Phim Viễn Tây", "Phim Cổ Trang", "Phim Nhạc", "Phim Lịch Sử", 
    "Phim Thần Thoại", "Phim Truyền Hình", "Chương Trình Truyền Hình", "Phim Khác"
]

# Tạo dictionary ánh xạ tự động
INTRO_TOPICS = {}
for i, genre in enumerate(GENRES_VI):
    # Lấy màu từ danh sách, lặp lại nếu cần
    color, gradient, hover_color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    INTRO_TOPICS[genre] = {
        "genres": [genre], # Ánh xạ trực tiếp 1-1
        "color": color, 
        "gradient": gradient,
        "hover_color": hover_color # Thêm màu hover
    }

# Lưu các thể loại duy nhất sau khi tiền xử lý
if 'ALL_UNIQUE_GENRES' not in st.session_state:
    st.session_state['ALL_UNIQUE_GENRES'] = [] 

# --- KHỞI TẠO BIẾN TRẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'
if 'is_new_user' not in st.session_state: # Thêm biến này từ code bạn cung cấp
    st.session_state['is_new_user'] = False
if 'user_genres' not in st.session_state: # Thêm biến này từ code bạn cung cấp
    st.session_state['user_genres'] = []
if 'guest_genres' not in st.session_state: # Thêm biến này từ code bạn cung cấp
    st.session_state['guest_genres'] = []
if 'selected_movie' not in st.session_state: # Thêm biến này từ code bạn cung cấp
    st.session_state['selected_movie'] = None
if 'last_results' not in st.session_state: # Thêm biến này từ code bạn cung cấp
    st.session_state['last_results'] = None

# Biến trạng thái cho kết quả và biểu đồ (Duy trì để tránh lỗi logic)
if 'last_sim_result' not in st.session_state: st.session_state['last_sim_result'] = pd.DataFrame()
if 'last_sim_movie' not in st.session_state: st.session_state['last_sim_movie'] = None
if 'show_sim_plot' not in st.session_state: st.session_state['show_sim_plot'] = False
if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False
if 'selected_intro_topics' not in st.session_state: st.session_state['selected_intro_topics'] = []
if 'last_guest_result' not in st.session_state: st.session_state['last_guest_result'] = pd.DataFrame()
if 'show_guest_plot' not in st.session_state: st.session_state['show_guest_plot'] = False
if 'selected_reg_topics' not in st.session_state: st.session_state['selected_reg_topics'] = set()


# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU & HELPERS
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """Hàm helper để tải dữ liệu CSV với cache."""
    try:
        df = pd.read_csv(file_path).fillna("")
        # Đảm bảo cột Năm phát hành tồn tại và là số
        if 'Năm phát hành' not in df.columns:
             df['Năm phát hành'] = pd.Timestamp('now').year
        return df
    except Exception as e:
        st.error(f"Lỗi tải file {file_path}: {e}")
        return pd.DataFrame()

def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)
    
def get_all_unique_genres(df_movies):
    all_genres = set()
    for genres_set in df_movies['parsed_genres']:
        all_genres.update(genres_set)
    return sorted(list(all_genres))

def safe_col(df, *cols):
    """Hàm an toàn để truy cập cột."""
    for c in cols:
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series([""] * len(df))

@st.cache_resource 
def load_and_preprocess_static_data():
    """Tải và tiền xử lý dữ liệu tĩnh (movies và mô hình)."""
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        if df_movies.empty: return pd.DataFrame(), np.array([[]]), None
        
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # 1. Tiền xử lý cho Content-Based (Dùng logic từ code mới của bạn)
        df_movies["content"] = (
            safe_col(df_movies, "Thể loại phim") + " " +
            safe_col(df_movies, "Diễn viên", "Diễn viên chính") + " " +
            safe_col(df_movies, "Đạo diễn")
        )

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["content"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến
        df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
        mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
        df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)
        
        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        # 3. Tiền xử lý cho Zero-Click
        if 'Năm phát hành' in df_movies.columns:
            df_movies['Năm phát hành'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(pd.Timestamp('now').year).astype(int)
            max_year = df_movies['Năm phát hành'].max()
            min_year = df_movies['Năm phát hành'].min()
            if max_year > min_year:
                 df_movies['recency_score'] = (df_movies['Năm phát hành'] - min_year) / (max_year - min_year)
            else:
                 df_movies['recency_score'] = 0.5 
        else:
            df_movies['recency_score'] = df_movies["popularity_norm"] * 0.1 

        # Tính global genre popularity score
        genres_pop = {}
        for index, row in df_movies.iterrows():
            popularity = row['Độ phổ biến']
            for genre in row['Thể loại phim'].split(','):
                genre = genre.strip()
                if genre:
                    genres_pop.setdefault(genre, []).append(popularity)
            
        global_genre_popularity = {g: sum(p)/len(p) for g, p in genres_pop.items() if len(p) > 0}
        max_pop = max(global_genre_popularity.values()) if global_genre_popularity else 1
        normalized_genre_pop = {g: p / max_pop for g, p in global_genre_popularity.items()}

        df_movies['global_genre_score'] = df_movies['Thể loại phim'].apply(
            lambda x: max([normalized_genre_pop.get(g.strip(), 0) for g in x.split(',')], default=0) if x else 0
        )
        
        ALL_GENRES = get_all_unique_genres(df_movies)
        st.session_state['ALL_UNIQUE_GENRES'] = ALL_GENRES
        return df_movies, cosine_sim_matrix, ALL_GENRES

    except Exception as e:
        st.error(f"LỖI TẢI HOẶC XỬ LÝ DỮ LIỆU TĨNH: {e}. Vui lòng kiểm tra các file CSV.")
        return pd.DataFrame(), np.array([[]]), None


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu người dùng vào Session State, đảm bảo các cột cần thiết tồn tại."""
    if 'df_users' not in st.session_state:
        REQUIRED_USER_COLUMNS = ['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất']
        
        try:
            # Kiểm tra file có tồn tại không
            if os.path.exists(USER_DATA_FILE):
                df_users = load_data(USER_DATA_FILE)
                df_users.columns = [col.strip() for col in df_users.columns]
            else:
                # Tạo DataFrame rỗng nếu file không tồn tại
                df_users = pd.DataFrame(columns=REQUIRED_USER_COLUMNS)

            # --- FIX CHO LỖI KEYERROR: Đảm bảo các cột cần thiết tồn tại ---
            for col in REQUIRED_USER_COLUMNS:
                if col not in df_users.columns:
                    df_users[col] = ""
            # -----------------------------------------------------------------
            
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
            df_users = df_users.dropna(subset=['ID'])
            
        except Exception:
            # Fallback nếu không thể tải file
            df_users = pd.DataFrame(columns=REQUIRED_USER_COLUMNS)

        st.session_state['df_users'] = df_users
    
    return st.session_state['df_users']

def get_unique_movie_titles(df_movies):
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

def set_auth_mode(mode):
    st.session_state['auth_mode'] = mode
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.session_state['selected_reg_topics'] = set() # Reset
    st.session_state['selected_intro_topics'] = []
    st.session_state['last_guest_result'] = pd.DataFrame()
    st.rerun()

def login_as_guest():
    st.session_state['logged_in_user'] = GUEST_USER
    st.session_state['auth_mode'] = 'login' 
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.session_state['selected_intro_topics'] = [] 
    st.session_state['last_guest_result'] = pd.DataFrame() 
    st.rerun()

def logout():
    st.session_state.clear() # Dùng clear() để xóa sạch state cũ theo yêu cầu trong code mới của bạn
    st.rerun()

# --- CALLBACK CHO GUEST MODE ---
def select_topic(topic_key):
    st.session_state['selected_intro_topics'] = [topic_key]
    st.session_state['last_guest_result'] = pd.DataFrame()
    st.rerun()

# --- CALLBACK CHO ĐĂNG KÝ (MỚI) ---
def toggle_reg_topic(topic):
    """Bật/Tắt chọn chủ đề trong lúc đăng ký"""
    if topic in st.session_state['selected_reg_topics']:
        st.session_state['selected_reg_topics'].remove(topic)
    else:
        st.session_state['selected_reg_topics'].add(topic)

# ------------------------------------------------------------------------------
# UI: CÁC HÀM VẼ GIAO DIỆN VÀ CSS (LIGHT THEME - BẮT MẮT)
# ------------------------------------------------------------------------------

def inject_light_theme():
    """Tiêm CSS để tạo giao diện Light Theme (Sáng, Tương phản cao, Bắt mắt)."""
    # Màu sắc chủ đạo Light Theme
    BG_COLOR = "#F7F9FC"      # Nền rất sáng
    CARD_BG = "#FFFFFF"       # Nền Card
    TEXT_COLOR = "#000000"    # Màu chữ tối (Đã đổi sang đen tuyệt đối)
    PRIMARY_COLOR = "#00BCD4" # Màu nhấn chính (Vibrant Cyan/Teal - Bắt mắt)
    SECONDARY_BG = "#E0F7FA"  # Sidebar/Input (Pale Cyan)
    ACCENT_COLOR = "#FF9800"  # Màu nhấn phụ (Vibrant Orange)
    DARK_TITLE_COLOR = "#1A1A1A" # Màu cho các tiêu đề (Gần đen)

    st.markdown(f"""
    <style>
        /* Tổng thể */
        .main, .stApp {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {SECONDARY_BG};
            color: {TEXT_COLOR};
            border-right: 2px solid {PRIMARY_COLOR}50;
        }}
        
        /* Header và Title */
        h1, h2, h3, h4, .st-emotion-cache-10trblm {{
            color: {DARK_TITLE_COLOR};
            font-weight: 800;
            text-shadow: 1px 1px 2px #AAAAAA50;
        }}
        
        /* Nút chung */
        .stButton button {{
            border-radius: 6px;
            padding: 10px 15px;
            font-weight: bold;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }}
        
        /* Nút Primary (Đăng nhập/Tìm kiếm) */
        .stButton button[kind="primary"] {{
            background-color: {PRIMARY_COLOR};
            color: {CARD_BG};
            border: 2px solid {PRIMARY_COLOR};
            box-shadow: 0 4px 10px {PRIMARY_COLOR}50;
        }}
        .stButton button[kind="primary"]:hover {{
            background-color: {ACCENT_COLOR}; /* Đổi màu khi hover */
            border-color: {ACCENT_COLOR};
            color: {CARD_BG};
            box-shadow: 0 4px 15px {ACCENT_COLOR}90;
        }}

        /* Nút Secondary (Auth Switch/Guest Button) */
        .stButton button[kind="secondary"] {{
            background-color: {CARD_BG};
            color: {TEXT_COLOR};
            border: 1px solid {PRIMARY_COLOR}50;
        }}
        .stButton button[kind="secondary"]:hover {{
            background-color: {SECONDARY_BG};
            border-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
        }}
        
        /* Info boxes */
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"] {{
            background-color: {SECONDARY_BG}AA;
            border-left: 5px solid {ACCENT_COLOR}; /* Điểm nhấn Orange */
            border-radius: 8px;
            padding: 10px;
            color: {TEXT_COLOR};
        }}
        
        /* Input fields */
        div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {{
            background-color: {CARD_BG};
            border-radius: 6px;
            color: {TEXT_COLOR};
            border: 1px solid #BBBBBB;
        }}

        /* --- CSS CHO CÁC THẺ (CARD) VÀ GRID (LIGHT LOOK) --- */
        div[data-testid*="stButton"] > button {{
             border: none; 
             transition: all 0.2s ease-in-out;
             color: white !important;
        }}

        /* Custom Grid Container */
        .movie-grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 25px;
            padding: 20px;
        }}
        
        /* Custom Movie Card Style */
        .movie-card {{
            background-color: {CARD_BG};
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); /* Shadow nhẹ trên nền sáng */
            transition: transform 0.3s, box-shadow 0.3s;
            height: 100%;
        }}
        .movie-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 10px 30px {PRIMARY_COLOR}50; /* Shadow Teal rực rỡ */
        }}
        .movie-poster {{
            width: 100%;
            height: 300px;
            background-color: {SECONDARY_BG}; /* Khối màu thay thế Poster */
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            border-bottom: 5px solid {PRIMARY_COLOR}; /* Dải màu nhấn */
        }}
        .movie-info {{
            padding: 15px;
        }}
        .movie-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {TEXT_COLOR};
        }}
        .movie-score {{
            font-size: 1.2rem;
            color: {PRIMARY_COLOR};
            font-weight: 800;
        }}
        .year-tag {{
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: {ACCENT_COLOR}; /* Màu Orange nổi bật */
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.8rem;
        }}
        .poster-text {{
            font-size: 1.2rem;
            color: {PRIMARY_COLOR};
            text-align: center;
            padding: 20px;
            font-weight: 600;
        }}
    </style>
    """, unsafe_allow_html=True)


def draw_registration_topic_cards():
    """Vẽ giao diện chọn chủ đề (Topic) với Light Theme."""
    
    st.markdown("### Bạn thích thể loại nào?")
    st.caption("Chọn các thể loại bạn thích để chúng tôi xây dựng hồ sơ ban đầu:")

    topics = list(INTRO_TOPICS.keys())
    # Giữ 4 cột cho các thẻ genre
    cols = st.columns(4) 
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        is_selected = topic in st.session_state['selected_reg_topics']
        
        # Style động: Nếu chọn thì có viền sáng/shadow
        border_style = "border: 3px solid #00BCD4;" if is_selected else "border: none;" # Màu nhấn Teal
        selected_shadow = "box-shadow: 0 0 18px rgba(0, 188, 212, 0.7);" if is_selected else "box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"
        opacity = "1.0" if is_selected else "0.9"
        
        # Tạo style riêng cho từng nút
        btn_style = f"""
            /* Base style - sử dụng gradient */
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 6px;
            height: 80px; 
            font-weight: bold;
            font-size: 0.95rem;
            width: 100%;
            margin-bottom: 8px;
            
            {border_style}
            {selected_shadow}
            opacity: {opacity};
            cursor: pointer;
            
            display: flex; 
            align-items: center; 
            justify-content: center;
            transition: all 0.2s ease-in-out;
        """
        
        # --- STYLE CHO HOVER MỚI: Đổi màu nền (dùng hover_color) ---
        hover_style = f"""
            div[data-testid="stButton"] button[key="reg_topic_{topic}"]:hover {{
                background: {data['hover_color']}; /* Đổi màu nền khi hover */
                transform: scale(1.03);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                border-color: #00BCD4 !important; /* Màu nhấn Teal khi hover */
                opacity: 1.0;
                color: white;
            }}
        """

        with cols[i % 4]:
            # Nút bấm toggle
            st.button(
                topic, 
                key=f"reg_topic_{topic}", 
                on_click=toggle_reg_topic, 
                args=(topic,),
                use_container_width=True
            )
            
            # Inject CSS chi tiết vào nút vừa tạo, bao gồm hover và active states
            st.markdown(f"""
                <style>
                    /* Style cơ bản */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"] {{
                        {btn_style}
                    }}
                    {hover_style}
                    /* Hiệu ứng ACTIVE/CLICK: nhấn chìm */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"]:active {{
                        transform: scale(0.98);
                        filter: brightness(90%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)


def draw_interest_cards_guest():
    """Giao diện thẻ cho chế độ Khách (Guest) - Chỉ chọn 1. LIGHT THEME."""
    st.header("Bạn đang quan tâm gì? ✨")
    st.markdown("Chọn **một** chủ đề để nhận đề xuất ngay lập tức:")
    
    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(4)
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        btn_style = f"""
            /* Base style - sử dụng gradient */
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 6px;
            height: 100px;
            font-weight: bold;
            font-size: 0.95rem;
            width: 100%;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease-in-out;
        """
        
        # --- STYLE CHO HOVER MỚI ---
        hover_style = f"""
            /* Hiệu ứng HOVER: Đổi sang màu solid/gradient khác */
            div[data-testid="stButton"] button[key="guest_{topic}"]:hover {{
                background: {data['hover_color']}; /* Đổi màu nền khi hover */
                transform: scale(1.03);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
                color: white;
            }}
        """

        with cols[i % 4]:
            st.button(topic, key=f"guest_{topic}", on_click=select_topic, args=(topic,), use_container_width=True)
            st.markdown(f"""
                <style>
                    div[data-testid="stButton"] button[key="guest_{topic}"] {{ 
                        {btn_style} 
                    }}
                    {hover_style}
                    /* Hiệu ứng ACTIVE/CLICK: nhấn chìm */
                    div[data-testid="stButton"] button[key="guest_{topic}"]:active {{
                        transform: scale(0.98);
                        filter: brightness(90%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)


def register_new_user_form(df_movies, cosine_sim):
    """Form đăng ký người dùng mới."""
    st.header("📝 Đăng Ký Tài Khoản Mới")
    st.info("📢 Người dùng mới sẽ chỉ tồn tại trong phiên làm việc hiện tại (Không lưu file CSV).")

    df_users = st.session_state['df_users']
    
    # 1. Nhập tên người dùng
    username = st.text_input("Tên người dùng mới (Duy nhất):", key="reg_username").strip()

    st.write("---")

    # 2. Chọn chủ đề
    draw_registration_topic_cards()
    
    selected_topics = list(st.session_state['selected_reg_topics'])
    
    st.write("")
    if selected_topics:
        st.success(f"✅ Đã chọn: {', '.join(selected_topics)}")
    else:
        st.warning("Vui lòng chọn ít nhất 1 thể loại.")

    st.write("---")

    # 3. Nút Đăng ký (Xử lý Logic lưu trữ)
    if st.button("🚀 Hoàn Tất Đăng Ký & Đăng Nhập", type="primary", use_container_width=True):
        if not username:
            st.error("Vui lòng nhập tên người dùng.")
            return
        
        if username in df_users['Tên người dùng'].values:
            st.error(f"❌ Tên người dùng '{username}' đã tồn tại.")
            return
        
        if not selected_topics:
            st.error("❌ Vui lòng chọn ít nhất 1 thể loại.")
            return
        
        # --- BƯỚC 1: XỬ LÝ DỮ LIỆU VÀ LƯU VÀO DF_USERS (TẠM) ---
        mapped_genres = set()
        for topic in selected_topics:
            if topic in INTRO_TOPICS:
                mapped_genres.update(INTRO_TOPICS[topic]['genres'])
            
        final_genres_list = list(mapped_genres)
        
        max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
        new_id = int(max_id) + 1
        
        # Cập nhật DataFrame người dùng
        new_user_data = {
            'ID': [new_id],
            'Tên người dùng': [username],
            '5 phim coi gần nhất': [str(final_genres_list)], 
            'Phim yêu thích nhất': [""] 
        }
        new_user_df = pd.DataFrame(new_user_data)
        st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
        
        st.session_state['logged_in_user'] = username
        
        # Cập nhật state cho luồng người dùng mới (dùng user_genres và is_new_user)
        st.session_state['user_genres'] = final_genres_list
        st.session_state['is_new_user'] = True
        
        # --- BƯỚC 2: TỰ ĐỘNG GỌI ĐỀ XUẤT HỒ SƠ VÀ LƯU VÀO SESSION STATE ---
        # Recommendations will be calculated on the next run when is_new_user is True

        st.balloons()
        st.success(f"🎉 Đăng ký thành công! Đã thiết lập hồ sơ theo sở thích: {', '.join(selected_topics)}.")
        
        # --- BƯỚC 3: CHẠY LẠI ỨNG DỤNG ĐỂ HIỂN THỊ KẾT QUẢ ĐỀ XUẤT ---
        st.rerun() 


def login_form():
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")
    df_users = st.session_state['df_users']

    with st.form("login_form"):
        username = st.text_input("Tên người dùng:").strip()
        submitted = st.form_submit_button("Đăng Nhập", type="primary", use_container_width=True)
        
        if submitted:
            if username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Đăng nhập thành công! Chào mừng, {username}.")
                st.rerun() 
            else:
                st.error("❌ Tên người dùng không tồn tại.")

def authentication_page(df_movies, cosine_sim):
    """Trang Xác thực."""
    
    # Inject Light Theme CSS
    inject_light_theme() 
    
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2, col3 = st.columns(3)
    
    # Nút Đăng nhập
    with col1:
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',), use_container_width=True, type="secondary")
    # Nút Đăng ký
    with col2:
        st.button("Đăng Ký", key="btn_register", on_click=set_auth_mode, args=('register',), use_container_width=True, type="secondary")
    # Nút Khách
    with col3:
        st.button("Khách 🚀", key="btn_guest_auth", on_click=login_as_guest, use_container_width=True, type="secondary")

    # Apply active style to the currently selected button
    # Sử dụng màu PRIMARY_COLOR (#00BCD4) cho trạng thái active
    if st.session_state['auth_mode'] == 'login':
        st.markdown("""<style>div[data-testid="column"] button[key="btn_login"] {background-color: #00BCD4 !important; border-color: #00BCD4 !important; color: white !important;}</style>""", unsafe_allow_html=True)
    elif st.session_state['auth_mode'] == 'register':
        st.markdown("""<style>div[data-testid="column"] button[key="btn_register"] {background-color: #00BCD4 !important; border-color: #00BCD4 !important; color: white !important;}</style>""", unsafe_allow_html=True)

    st.write("---")
    
    if st.session_state['auth_mode'] == 'login':
        login_form()
    
    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies, cosine_sim)

# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ
# ==============================================================================

# Tạo danh sách màu sắc rực rỡ và dễ phân biệt
def get_vibrant_colors(n):
    """Tạo n màu sắc phù hợp với Light Theme."""
    # Dùng colormap 'tab20' hoặc 'Set1' để có màu nổi bật trên nền sáng
    cmap = plt.cm.get_cmap('tab20', n)
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]
    return colors

def plot_recommendation_comparison(df_results, recommendation_type, movie_name=None):
    """
    Vẽ biểu đồ so sánh điểm số đề xuất (hoặc độ phổ biến) của các phim.
    Thiết lập cho Light Theme.
    """
    if df_results.empty:
        st.warning("Không có dữ liệu để vẽ biểu đồ.")
        return

    # 1. Xác định Cột điểm và Tiêu đề
    if 'weighted_score' in df_results.columns:
        score_col = 'weighted_score'
        y_label = "Điểm Đề xuất Tổng hợp"
        title_prefix = f"So sánh Đề xuất theo Tên Phim ('{movie_name}')"
    elif 'Similarity_Score' in df_results.columns:
        score_col = 'Similarity_Score'
        y_label = "Điểm Giống nhau (Genre Match)"
        title_prefix = f"So sánh Đề xuất theo AI (Genre Score)"
    elif 'combined_zero_click_score' in df_results.columns:
        score_col = 'combined_zero_click_score'
        y_label = "Điểm Zero-Click (Trend + Genre Boost)"
        title_prefix = "So sánh Đề xuất Zero-Click"
    else:
        score_col = 'Độ phổ biến'
        y_label = "Độ Phổ Biến"
        title_prefix = "So sánh Độ Phổ Biến"
        
    title = f"{title_prefix}\n({recommendation_type})"

    df_plot = df_results.sort_values(by=score_col, ascending=True).copy()
    
    num_movies = len(df_plot)
    colors = get_vibrant_colors(num_movies)

    # Cấu hình Light Theme cho Matplotlib
    BG_COLOR_MPL = "#F7F9FC" 
    TEXT_COLOR_MPL = "#333333"
    
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    ax.set_facecolor(BG_COLOR_MPL)
    fig.patch.set_facecolor(BG_COLOR_MPL)
    
    bars = ax.bar(df_plot['Tên phim'], df_plot[score_col], 
                      color=colors, edgecolor='#333333', alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + ax.get_ylim()[1]*0.01, 
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold', rotation=45, color=TEXT_COLOR_MPL)

    # Thiết lập màu sắc và font cho biểu đồ
    ax.set_title(title, fontsize=14, color='#00BCD4') # Màu nhấn Teal
    ax.set_xlabel("Tên Phim", color=TEXT_COLOR_MPL)
    ax.set_ylabel(y_label, color=TEXT_COLOR_MPL)
    ax.tick_params(axis='x', colors=TEXT_COLOR_MPL)
    ax.tick_params(axis='y', colors=TEXT_COLOR_MPL)
    ax.spines['left'].set_color(TEXT_COLOR_MPL)
    ax.spines['bottom'].set_color(TEXT_COLOR_MPL)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    
    plt.tight_layout()
    st.pyplot(fig)


def get_zero_click_recommendations(df_movies, selected_genres, num_recommendations=15):
    WEIGHT_POPULARITY = 0.50 
    WEIGHT_RECENCY = 0.25
    WEIGHT_GENRE_POPULARITY = 0.25
    WEIGHT_TOPIC_BOOST = 0.50 
    
    if df_movies.empty or 'popularity_norm' not in df_movies.columns: return pd.DataFrame()
    df = df_movies.copy()
    
    df['base_zero_click_score'] = (
        WEIGHT_POPULARITY * df['popularity_norm'] +
        WEIGHT_RECENCY * df['recency_score'] +
        WEIGHT_GENRE_POPULARITY * df['global_genre_score']
    )
    
    if selected_genres:
        def calculate_boost(parsed_genres):
            return 1 if not parsed_genres.isdisjoint(set(selected_genres)) else 0
        df['topic_boost'] = df['parsed_genres'].apply(calculate_boost)
        df['combined_zero_click_score'] = df['base_zero_click_score'] + (df['topic_boost'] * WEIGHT_TOPIC_BOOST)
    else:
        df['combined_zero_click_score'] = df['base_zero_click_score']

    recommended_df = df.sort_values(by='combined_zero_click_score', ascending=False)
    # Bao gồm Năm phát hành cho hiển thị Card
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Năm phát hành', 'combined_zero_click_score']].head(num_recommendations)


def get_recommendations(username, df_movies, num_recommendations=10):
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    # FIX LỖI: Sử dụng .values[0] để truy cập giá trị string an toàn
    user_genres_str = user_row['5 phim coi gần nhất'].values[0]
    user_genres_list = []
    
    try:
        # Giả định user_genres_str là một chuỗi của list Python (vd: "['Action', 'Drama']")
        user_genres_list = ast.literal_eval(user_genres_str)
        if not isinstance(user_genres_list, list): user_genres_list = []
    except (ValueError, SyntaxError):
        # Fallback cho trường hợp chuỗi không đúng định dạng list Python, thử phân tách bằng dấu phẩy
        watched_list = [m.strip().strip("'") for m in user_genres_str.strip('[]').split(',') if m.strip()]
        watched_genres_df = df_movies[df_movies['Tên phim'].isin(watched_list)]
        user_genres_list = []
        for genres in watched_genres_df['parsed_genres']:
            user_genres_list.extend(genres)
        
    user_genres = set(user_genres_list)
    
    # Lấy phim yêu thích (nếu có) để boost thêm - Dùng .values[0]
    favorite_movie = user_row['Phim yêu thích nhất'].values[0]
    if favorite_movie:
        favorite_movie_genres = df_movies[df_movies['Tên phim'] == favorite_movie]['parsed_genres'].iloc[0] if not df_movies[df_movies['Tên phim'] == favorite_movie].empty else set()
        user_genres.update(favorite_movie_genres)

    if not user_genres: return pd.DataFrame()

    candidate_movies = df_movies[df_movies['Tên phim'] != favorite_movie].copy()
    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(lambda x: len(x.intersection(user_genres)))

    recommended_df = candidate_movies.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False])
    # Bao gồm Năm phát hành cho hiển thị Card
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Năm phát hành', 'Similarity_Score']].head(num_recommendations)

def get_movie_index(movie_name, df_movies):
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
        return idx
    except IndexError:
        return -1

def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    if cosine_sim.size == 0 or df_movies.empty: return pd.DataFrame()
    idx = get_movie_index(movie_name, df_movies)
    if idx == -1: return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])
    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    df_result['weighted_score'] = (weight_sim * df_result['similarity'] + weight_pop * df_result['popularity_norm'])
    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)
    # Bao gồm Năm phát hành cho hiển thị Card
    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Năm phát hành', 'Thể loại phim']].head(10)

def display_movie_grid(df_results, score_column):
    """Hiển thị kết quả đề xuất dưới dạng lưới phim (movie grid) với Poster Placeholder."""
    
    if df_results.empty:
        st.warning("Không có phim nào để hiển thị.")
        return

    # Khởi tạo HTML cho lưới
    grid_html = '<div class="movie-grid-container">'
    
    for index, row in df_results.iterrows():
        title = row['Tên phim']
        # Sửa lỗi: Đảm bảo score_column tồn tại trước khi truy cập
        score = row.get(score_column, 0)
        # Xử lý Năm phát hành, đảm bảo là số nguyên
        year = int(row.get('Năm phát hành', 'N/A')) if pd.notna(row.get('Năm phát hành')) and row.get('Năm phát hành') != "" else 'N/A'
        
        # Placeholder Text
        # Sử dụng ký tự Unicode (🎬) để trang trí thay vì hình ảnh
        poster_text = "🎬 Phim đề xuất"

        
        # Dùng Score làm điểm hiển thị chính
        score_display = f"ĐIỂM: {score:.2f}" if isinstance(score, (int, float)) else "N/A"
        
        card_html = f"""
        <div class="movie-card">
            <div class="movie-poster">
                <div class="poster-text">{poster_text}</div>
                <span class="year-tag">{year}</span>
            </div>
            <div class="movie-info">
                <div class="movie-title" title="{title}">{title}</div>
                <div class="movie-score">{score_display}</div>
            </div>
        </div>
        """
        grid_html += card_html
        
    grid_html += '</div>'
    
    st.markdown(grid_html, unsafe_allow_html=True)

# ======================================================
# 📊 USER TREND CHART (TỪ CODE MỚI CỦA BẠN)
# ======================================================
def plot_user_trend_from_movies(df_movies, movie_list, username):
    genres = []
    # Lấy genres từ DataFrame movies_df dựa trên danh sách tên phim
    for m in movie_list:
        row = df_movies[df_movies["Tên phim"] == m]
        if not row.empty:
            # Lấy chuỗi thể loại, tách và thêm vào list genres
            genres.extend(
                str(row.iloc[0]["Thể loại phim"]).split(",")
            )

    # Làm sạch khoảng trắng
    clean_genres = [g.strip() for g in genres if g.strip()]
    
    if not clean_genres:
        st.info("Không đủ dữ liệu để vẽ biểu đồ xu hướng.")
        return

    counter = Counter(clean_genres)
    labels = list(counter.keys())
    values = list(counter.values())

    # Cấu hình Light Theme cho Matplotlib
    BG_COLOR_MPL = "#F7F9FC" 
    TEXT_COLOR_MPL = "#333333"
    ACCENT_COLOR_MPL = "#FF9800" # Orange
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(labels, values, color=ACCENT_COLOR_MPL, alpha=0.8)
    ax.set_title(f"Xu hướng xem phim của {username}", color='#1A1A1A', fontsize=14, fontweight='bold')
    ax.set_ylabel("Số lần", color=TEXT_COLOR_MPL)
    ax.set_xlabel("Thể loại", color=TEXT_COLOR_MPL)
    
    # Theme màu sáng
    ax.set_facecolor(BG_COLOR_MPL)
    fig.patch.set_facecolor(BG_COLOR_MPL)
    ax.tick_params(axis='x', colors=TEXT_COLOR_MPL)
    ax.tick_params(axis='y', colors=TEXT_COLOR_MPL)
    ax.spines['left'].set_color(TEXT_COLOR_MPL)
    ax.spines['bottom'].set_color(TEXT_COLOR_MPL)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# ==============================================================================
# V. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    
    # Inject Light Theme CSS
    inject_light_theme() 
    
    df_users = st.session_state['df_users']
    is_guest = st.session_state.get('logged_in_user') == GUEST_USER
    username = st.session_state.get('logged_in_user')
    username_display = "Khách" if is_guest else username

    
    # ======================================================
    # DETAIL PAGE LOGIC
    # ======================================================
    if st.session_state.selected_movie:
        # Lấy row chi tiết của phim được chọn
        m_row = df_movies[df_movies["Tên phim"] == st.session_state.selected_movie].iloc[0]
        
        # Hiển thị chi tiết phim
        st.markdown(f"""
        <div style='background-color: #E0F7FA; padding: 40px; border-radius: 10px; border: 2px solid #00BCD4; margin-bottom: 20px; text-align: center;'>
            <h1 style='color: #00BCD4;'>{m_row["Tên phim"]}</h1>
            <p style='color: #1A1A1A;'>Đạo diễn: {m_row.get("Đạo diễn", "N/A")}</p>
            <p style='color: #1A1A1A;'>Diễn viên chính: {m_row.get("Diễn viên chính", "N/A")}</p>
            <p style='color: #1A1A1A;'>Thể loại: 🎭 {m_row.get("Thể loại phim", "N/A")}</p>
            <p style='color: #1A1A1A;'>Mô tả: {m_row.get("Mô tả", "N/A")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎯 Phim tương tự (Content-Based)")
        # Lấy 5 phim tương tự (dùng content_based)
        recommended_df = content_based(m_row["Tên phim"], cosine_sim, df_movies, 5)
        display_movie_grid(recommended_df, 'score')
        
        if st.button("⬅️ Quay lại"):
            st.session_state.selected_movie = None
            st.rerun()
        return

    # ======================================================
    # SIDEBAR MENU
    # ======================================================
    menu = st.sidebar.radio(
        "Menu",
        [
            "Cá nhân",
            "Đề xuất theo Tên Phim",
            "Đề xuất theo AI",
            "Đề xuất theo Thể loại Yêu thích",
            "Đăng Xuất"
        ]
    )

    if menu == "Đăng Xuất":
        logout() # Dùng hàm logout đã được chỉnh sửa để clear state

    # ======================================================
    # HOME CONTENT
    # ======================================================
    st.header(f"🎬 Chào mừng, {username_display}")

    # ================== 👤 CÁ NHÂN ==================
    if menu == "Cá nhân":
        if is_guest:
            st.info("Chế độ Khách không có trang Cá nhân.")
        else:
            user = df_users[
                df_users["Tên người dùng"] == username
            ].iloc[0]

            st.subheader("🎞️ 5 phim bạn xem gần nhất")

            try:
                # Sửa lỗi: Lấy danh sách tên phim từ cột
                recent_movies_str = user["5 phim coi gần nhất"]
                # Cột này lưu list genres dưới dạng chuỗi khi đăng ký, nếu là user cũ có thể là tên phim
                
                # Cố gắng chuyển đổi thành list tên phim (nếu có)
                if recent_movies_str.startswith("[") and recent_movies_str.endswith("]"):
                     # Trường hợp user mới, cột này lưu list genres. Cần tìm 5 phim ngẫu nhiên thuộc genres đó
                     # Tạm thời chỉ lấy tên phim nếu cột này chứa tên phim
                     if any(m in df_movies['Tên phim'].values for m in ast.literal_eval(recent_movies_str)):
                         recent_movies_list = ast.literal_eval(recent_movies_str)
                     else:
                          # Nếu là list genres, ta hiển thị 5 phim ngẫu nhiên thuộc các genres đó
                          genres = ast.literal_eval(recent_movies_str)
                          recent_df = recommend_by_genres(df_movies, genres, top_n=5)
                          recent_movies_list = recent_df['Tên phim'].tolist()

                else:
                    recent_movies_list = [recent_movies_str] # Chỉ có 1 tên phim? Rất khó xử lý
                    
            except Exception as e:
                # Nếu không thể parse, giả định cột này chứa tên phim
                if recent_movies_str and not recent_movies_str.startswith("["):
                    recent_movies_list = [recent_movies_str] 
                else:
                    recent_movies_list = []
            
            # --- LOGIC MỚI: DÙNG HÀM TRUY VẤN ĐỂ LẤY DF CHI TIẾT ---
            # Đây là logic chuẩn để lấy DF từ list tên phim
            if recent_movies_list and recent_movies_list[0]:
                recent_df = df_movies[df_movies["Tên phim"].isin(recent_movies_list)]
            else:
                recent_df = pd.DataFrame()
            
            # Đảm bảo hiển thị 5 phim nếu có
            if not recent_df.empty:
                display_movie_grid(recent_df, 'Độ phổ biến') # Dùng Độ phổ biến làm score tạm
            else:
                st.info("Chưa có phim nào trong lịch sử xem của bạn.")

            st.subheader("📊 Xu hướng xem phim")
            plot_user_trend_from_movies(df_movies, recent_movies_list, username)


    # ================== USER MỚI (CHUYỂN HƯỚNG TỪ ĐĂNG KÝ) ==================
    elif st.session_state.is_new_user:
        st.subheader("🎯 Thiết lập Hồ sơ & Đề xuất ban đầu")
        
        # Giữ lại logic cho phép người dùng điều chỉnh thể loại
        st.session_state.user_genres = st.multiselect(
            "Thể loại muốn xem:",
            st.session_state['ALL_UNIQUE_GENRES'],
            default=st.session_state.user_genres
        )

        if st.button("🎬 Đề xuất phim", type="primary"):
            st.session_state.last_results = recommend_by_genres(
                df_movies, st.session_state.user_genres
            )
            st.session_state.is_new_user = False # Kết thúc quá trình thiết lập ban đầu
            st.rerun()
        
        # Chạy đề xuất lần đầu (nếu chưa có kết quả)
        if st.session_state.last_results is None:
             st.session_state.last_results = recommend_by_genres(
                df_movies, st.session_state.user_genres, top_n=10
            )

    # ================== CONTENT BASED ==================
    elif menu == "Đề xuất theo Tên Phim":
        st.subheader("1️⃣ Đề xuất theo Nội dung")
        movie_titles = df_movies["Tên phim"].unique().tolist()
        
        # selectbox chọn tên phim
        selected_movie_name = st.selectbox("Chọn phim:", movie_titles)
        
        # Sửa lỗi: Khi bấm nút "Tìm" sẽ hiển thị chính phim đó và kích hoạt trang chi tiết
        if st.button("🔍 Tìm kiếm & Xem chi tiết", type="primary"):
            st.session_state.selected_movie = selected_movie_name # Chuyển sang trang chi tiết
            st.rerun()
        
        # Nếu đã có kết quả trước đó, hiển thị kết quả (Content Based của phim đã chọn trước đó)
        if st.session_state.last_results is not None:
             st.markdown("---")
             st.subheader("Phim tương tự lần cuối:")
             # Hiển thị kết quả dưới dạng grid
             display_movie_grid(st.session_state.last_results, 'score')


    # ================== AI + REFRESH (PROFILE BASED) ==================
    elif menu == "Đề xuất theo AI":
        st.subheader("2️⃣ Đề xuất theo AI (Profile Based)")
        
        if st.button("🎬 Đề xuất AI", type="primary"):
            if is_guest:
                st.session_state.last_results = recommend_by_genres(
                    df_movies, st.session_state.guest_genres, top_n=10
                )
            else:
                user = df_users[df_users["Tên người dùng"] == username].iloc[0]
                st.session_state.last_results = profile_based(df_movies, user, top_n=10)
            st.rerun()

        if st.button("🔄 Tạo đề xuất mới"):
            # Logic tạo đề xuất mới giống hệt nút chính, chỉ khác là không đổi màu
            if is_guest:
                st.session_state.last_results = recommend_by_genres(
                    df_movies, st.session_state.guest_genres, top_n=10
                )
            else:
                user = df_users[df_users["Tên người dùng"] == username].iloc[0]
                st.session_state.last_results = profile_based(df_movies, user, top_n=10)
            st.rerun()

    # ================== GENRE FAVORITE ==================
    elif menu == "Đề xuất theo Thể loại Yêu thích":
        st.subheader("3️⃣ Đề xuất theo Thể loại Yêu thích (Favorite Genre)")
        
        if is_guest:
            st.session_state.last_results = recommend_by_genres(
                df_movies, st.session_state.guest_genres, top_n=10
            )
            if st.button("🔄 Tạo đề xuất mới", type="primary"):
                st.session_state.last_results = recommend_by_genres(
                    df_movies, st.session_state.guest_genres, top_n=10
                )
                st.rerun()
        else:
            user = df_users[df_users["Tên người dùng"] == username].iloc[0]
            fav = user["Phim yêu thích nhất"]
            
            if fav and fav in df_movies["Tên phim"].values:
                # Lấy thể loại của phim yêu thích nhất
                genres_row = df_movies[df_movies["Tên phim"] == fav]["Thể loại phim"].values[0]
                g = [x.strip() for x in genres_row.split(",") if x.strip()]
                
                st.info(f"Đang đề xuất dựa trên phim yêu thích nhất của bạn: **{fav}** (Thể loại: {', '.join(g)})")
                
                if st.button("🎬 Đề xuất theo phim yêu thích", type="primary"):
                    st.session_state.last_results = recommend_by_genres(df_movies, g, top_n=10)
                    st.rerun()

            else:
                 st.warning("Bạn chưa có phim yêu thích nhất hoặc phim đó không tồn tại trong CSDL.")
                 # Chuyển sang profile based nếu không tìm thấy phim yêu thích nhất
                 if st.button("Đề xuất bằng AI Profile thay thế", type="secondary"):
                    st.session_state.last_results = profile_based(df_movies, user, top_n=10)
                    st.rerun()


    # ======================================================
    # SHOW RESULTS (FOR AI AND GENRE MENUS)
    # ======================================================
    if st.session_state.last_results is not None and menu not in ["Cá nhân", "Đề xuất theo Tên Phim"]:
        st.markdown("---")
        st.subheader("Kết quả đề xuất:")
        # Hiển thị kết quả dưới dạng grid
        display_movie_grid(st.session_state.last_results, 'score') # Giả định cột score/similarity score

        # Thêm tùy chọn hiển thị biểu đồ nếu cần
        if st.checkbox("📊 Hiển thị Biểu đồ Xu hướng Phim Đề Xuất", key="show_rec_plot"):
             # Cần tính lại genres từ df_results
             rec_genres = st.session_state.last_results['Thể loại phim'].str.split(",").explode().tolist()
             plot_user_trend_from_movies(df_movies, rec_genres, "Phim Đề Xuất")
             
        with st.expander("Xem chi tiết dưới dạng bảng"):
            st.dataframe(st.session_state.last_results, use_container_width=True)

# ==============================================================================
# WRAPPER TO RUN APP
# ==============================================================================
if __name__ == '__main__':
    
    # Tải dữ liệu và tính toán
    movies_df, cosine_sim, ALL_GENRES = load_and_preprocess_static_data()
    users_df = initialize_user_data()

    # Kiểm tra df rỗng
    if movies_df.empty:
        st.error("Không thể tải dữ liệu phim.")
        sys.exit()
    
    # Bổ sung logic Content Based vào các hàm đề xuất chung
    def content_based(movie_name, cosine_sim_matrix, df_movies, top_n=10):
        if movie_name not in df_movies["Tên phim"].values:
            return df_movies.sample(top_n)
        idx = df_movies[df_movies["Tên phim"] == movie_name].index[0]
        scores = list(enumerate(cosine_sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        # Thêm cột score tạm thời cho display_movie_grid
        result_df = df_movies.iloc[[i[0] for i in scores]].copy()
        result_df['score'] = [i[1] for i in scores]
        return result_df

    def recommend_by_genres(df_movies, genres, top_n=10):
        df_movies['match_score'] = df_movies['parsed_genres'].apply(lambda x: len(x.intersection(set(genres))))
        df = df_movies[df_movies['match_score'] > 0].sort_values(by=['match_score', 'Độ phổ biến'], ascending=[False, False])
        
        # Thêm cột score tạm thời cho display_movie_grid
        result_df = df.head(top_n).copy()
        result_df['score'] = result_df['match_score']
        return result_df

    def profile_based(df_movies, user_row, top_n=10):
        # Lấy danh sách phim đã xem (đã có ở dạng chuỗi list genres khi đăng ký)
        user_genres_str = user_row['5 phim coi gần nhất']
        user_genres_list = []
        try:
            # Sửa lỗi: Cột này lưu list genres dưới dạng chuỗi khi đăng ký
            user_genres_list = ast.literal_eval(user_genres_str)
        except:
            pass # Mặc định là list rỗng
        
        if not user_genres_list:
             # Nếu user mới, chỉ dựa vào độ phổ biến chung
             df = df_movies.sort_values(by=['Độ phổ biến'], ascending=False).head(top_n).copy()
             df['score'] = df['Độ phổ biến'] # score là độ phổ biến
             return df
        
        # Lấy thể loại phổ biến nhất
        clean_genres = [g.strip() for g in user_genres_list if g.strip()]
        if not clean_genres:
             return df_movies.sample(top_n).copy().assign(score=1) # Fallback
             
        # Dùng logic recommend_by_genres với genres của user
        return recommend_by_genres(df_movies, clean_genres, top_n)


    main_page(movies_df, cosine_sim)
