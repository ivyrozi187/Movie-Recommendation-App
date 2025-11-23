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

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" 

# --- CẤU HÌNH DANH SÁCH THỂ LOẠI (TOPICS) THEO YÊU CẦU ---
# Danh sách màu sắc cho Dark Theme
COLOR_PALETTE = [
    ("#FF4500", "#FF6347", "#CC3700"), # OrangeRed (Action)
    ("#1E90FF", "#4169E1", "#1773CC"), # DodgerBlue (Sci-Fi)
    ("#3CB371", "#66CDAA", "#309C60"), # MediumSeaGreen (Comedy)
    ("#800080", "#BA55D3", "#660066"), # Purple (Fantasy)
    ("#FFD700", "#FFA500", "#CCAA00"), # Gold (Adventure)
    ("#F08080", "#CD5C5C", "#C86060"), # LightCoral (Drama)
    ("#00CED1", "#20B2AA", "#00A5A8"), # DarkTurquoise (Thriller)
    ("#FF69B4", "#FFC0CB", "#CC5090"), # HotPink (Romance)
    ("#D2B48C", "#F5DEB3", "#B39572"), # Tan (History)
    ("#B0C4DE", "#778899", "#89A0B8"), # LightSteelBlue (Mystery)
    ("#6A5ACD", "#8470FF", "#5446A4"), # SlateBlue (Crime)
    ("#5F9EA0", "#87CEEB", "#4C7F80"), # CadetBlue (Western)
    ("#D8A4E6", "#C780D3", "#B370C0"), # Muted Lavender
    ("#FF8C00", "#FFA040", "#CC7000"), # Dark Orange
    ("#8B0000", "#A52A2A", "#680000"), # Dark Red
    ("#483D8B", "#6A5ACD", "#3A316E"), # Dark Slate Blue
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

# Biến trạng thái cho kết quả và biểu đồ
if 'last_sim_result' not in st.session_state: st.session_state['last_sim_result'] = pd.DataFrame()
if 'last_sim_movie' not in st.session_state: st.session_state['last_sim_movie'] = None
if 'show_sim_plot' not in st.session_state: st.session_state['show_sim_plot'] = False

if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False

# Biến trạng thái cho Guest Mode
if 'selected_intro_topics' not in st.session_state: st.session_state['selected_intro_topics'] = []
if 'last_guest_result' not in st.session_state: st.session_state['last_guest_result'] = pd.DataFrame()
if 'show_guest_plot' not in st.session_state: st.session_state['show_guest_plot'] = False

# --- BIẾN TRẠNG THÁI MỚI CHO ĐĂNG KÝ (TOPICS) ---
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

@st.cache_resource 
def load_and_preprocess_static_data():
    """Tải và tiền xử lý dữ liệu tĩnh (movies và mô hình)."""
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        if df_movies.empty: return pd.DataFrame(), np.array([[]])
        
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # 1. Tiền xử lý cho Content-Based
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
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
        
        st.session_state['ALL_UNIQUE_GENRES'] = get_all_unique_genres(df_movies)
        return df_movies, cosine_sim_matrix 

    except Exception as e:
        st.error(f"LỖI TẢI HOẶC XỬ LÝ DỮ LIỆU TĨNH: {e}. Vui lòng kiểm tra các file CSV.")
        return pd.DataFrame(), np.array([[]])


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
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.session_state['selected_intro_topics'] = []
    st.session_state['last_guest_result'] = pd.DataFrame() 
    st.session_state['selected_reg_topics'] = set()
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
# UI: CÁC HÀM VẼ GIAO DIỆN VÀ CSS (DARK THEME - RO PHIM STYLE)
# ------------------------------------------------------------------------------

def inject_dark_theme():
    """Tiêm CSS để tạo giao diện Dark Theme (Phong cách Rophim)."""
    # Màu sắc chủ đạo Dark Theme
    BG_COLOR = "#0F1113"      # Nền rất tối (Gần đen)
    CARD_BG = "#1A1D20"       # Nền Card/Dashboard
    TEXT_COLOR = "#E0E0E0"    # Màu chữ sáng
    PRIMARY_COLOR = "#FF4500" # Màu cam nhấn (OrangeRed)
    SECONDARY_BG = "#2B2D30"  # Sidebar và background phụ
    ACCENT_COLOR = "#1E90FF"  # Màu nhấn phụ (Blue)

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
            border-right: 2px solid {PRIMARY_COLOR}50; /* Viền mỏng */
        }}
        
        /* Header và Title */
        h1, h2, h3, h4, .st-emotion-cache-10trblm {{ /* Lớp chứa tiêu đề */
            color: {PRIMARY_COLOR};
            font-weight: 700;
        }}
        
        /* Nút chính (Đăng ký/Tìm kiếm) */
        .stButton button {{
            border-radius: 6px;
            padding: 10px 15px;
            font-weight: bold;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }}
        
        /* Nút Primary */
        .stButton button[kind="primary"] {{
            background-color: {PRIMARY_COLOR};
            color: {BG_COLOR};
            border: 2px solid {PRIMARY_COLOR};
        }}
        .stButton button[kind="primary"]:hover {{
            background-color: {ACCENT_COLOR}; /* Đổi màu khi hover */
            border-color: {ACCENT_COLOR};
            color: white;
        }}

        /* Nút Secondary (Login/Register) */
        .stButton button[kind="secondary"] {{
            background-color: {SECONDARY_BG};
            color: {TEXT_COLOR};
            border: 1px solid {SECONDARY_BG};
        }}
        .stButton button[kind="secondary"]:hover {{
            background-color: {ACCENT_COLOR}30;
            border-color: {ACCENT_COLOR};
            color: {TEXT_COLOR};
        }}
        

        /* Info boxes */
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"] {{
            background-color: {CARD_BG}AA; /* Nền card mờ */
            border-left: 5px solid {ACCENT_COLOR};
            border-radius: 8px;
            padding: 10px;
            color: {TEXT_COLOR};
        }}
        
        /* Dataframe */
        .stDataFrame {{
            background-color: {CARD_BG};
        }}

        /* Input fields */
        div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {{
            background-color: {SECONDARY_BG};
            border-radius: 6px;
            color: {TEXT_COLOR};
        }}

        /* --- CSS CHO CÁC THẺ (CARD) VÀ GRID --- */
        /* CSS cho nút chọn thể loại trong form đăng ký */
        div[data-testid*="stButton"] > button {{
             border: none; 
             transition: all 0.2s ease-in-out;
             color: white !important;
        }}

        /* Custom Grid Container */
        .movie-grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 20px;
            padding: 10px;
        }}
        
        /* Custom Movie Card Style */
        .movie-card {{
            background-color: {CARD_BG};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s, box-shadow 0.3s;
            height: 100%; /* Giữ chiều cao linh hoạt trong grid item */
        }}
        .movie-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(255, 69, 0, 0.4); /* Shadow cam nổi bật */
        }}
        .movie-poster {{
            width: 100%;
            height: 250px; /* Chiều cao cố định cho poster */
            background-color: {SECONDARY_BG};
            display: flex;
            align-items: center;
            justify-content: center;
            color: {TEXT_COLOR}80;
            font-size: 14px;
            font-weight: 500;
            position: relative; /* Dùng cho Year tag */
        }}
        .movie-info {{
            padding: 10px;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .movie-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {TEXT_COLOR};
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }}
        .movie-score {{
            font-size: 0.9rem;
            color: {PRIMARY_COLOR};
            font-weight: bold;
            margin-top: 5px;
        }}
        .year-tag {{
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: {PRIMARY_COLOR}E0;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.8rem;
        }}
    </style>
    """, unsafe_allow_html=True)


def draw_registration_topic_cards():
    """Vẽ giao diện chọn chủ đề (Topic) thay vì chọn từng genre lẻ. DARK THEME."""
    
    st.markdown("### Bạn thích thể loại nào?")
    st.caption("Chọn các thể loại bạn thích để chúng tôi xây dựng hồ sơ ban đầu:")

    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(4) 
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        is_selected = topic in st.session_state['selected_reg_topics']
        
        # Style động: Nếu chọn thì có viền sáng/shadow
        border_style = "border: 3px solid #FF4500;" if is_selected else "border: none;" # Màu nhấn Cam
        selected_shadow = "box-shadow: 0 0 18px rgba(255, 69, 0, 0.7);" if is_selected else "box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);"
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
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
                border-color: #FF4500 !important; /* Màu nhấn Cam khi hover */
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
    """Giao diện thẻ cho chế độ Khách (Guest) - Chỉ chọn 1. DARK THEME."""
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
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            transition: all 0.2s ease-in-out;
        """
        
        # --- STYLE CHO HOVER MỚI ---
        hover_style = f"""
            /* Hiệu ứng HOVER: Đổi sang màu solid/gradient khác */
            div[data-testid="stButton"] button[key="guest_{topic}"]:hover {{
                background: {data['hover_color']}; /* Đổi màu nền khi hover */
                transform: scale(1.03);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
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
        
        # --- BƯỚC 2: TỰ ĐỘNG GỌI ĐỀ XUẤT HỒ SƠ VÀ LƯU VÀO SESSION STATE ---
        recommendations = get_recommendations(username, df_movies)

        if not recommendations.empty:
            st.session_state['last_profile_recommendations'] = recommendations
            st.session_state['show_profile_plot'] = True
        else:
            st.session_state['last_profile_recommendations'] = pd.DataFrame()
            st.session_state['show_profile_plot'] = False

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
    
    # Inject Dark Theme CSS
    inject_dark_theme() 
    
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2, col3 = st.columns(3)
    
    # Nút Đăng nhập
    with col1:
        # Dùng type="primary" cho nút active, dùng custom CSS để tô màu active
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',), use_container_width=True, type="secondary")
    # Nút Đăng ký
    with col2:
        st.button("Đăng Ký", key="btn_register", on_click=set_auth_mode, args=('register',), use_container_width=True, type="secondary")
    # Nút Khách
    with col3:
        st.button("Khách 🚀", key="btn_guest_auth", on_click=login_as_guest, use_container_width=True, type="secondary")

    # Apply active style to the currently selected button
    if st.session_state['auth_mode'] == 'login':
        st.markdown("""<style>div[data-testid="column"] button[key="btn_login"] {background-color: #FF4500 !important; border-color: #FF4500 !important; color: #0F1113 !important;}</style>""", unsafe_allow_html=True)
    elif st.session_state['auth_mode'] == 'register':
        st.markdown("""<style>div[data-testid="column"] button[key="btn_register"] {background-color: #FF4500 !important; border-color: #FF4500 !important; color: #0F1113 !important;}</style>""", unsafe_allow_html=True)

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
    """Tạo n màu sắc phù hợp với Dark Theme."""
    # Dùng colormap 'Spectral' hoặc 'nipy_spectral'
    cmap = plt.cm.get_cmap('Spectral', n)
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]
    return colors

def plot_recommendation_comparison(df_results, recommendation_type, movie_name=None):
    """
    Vẽ biểu đồ so sánh điểm số đề xuất (hoặc độ phổ biến) của các phim.
    Thiết lập cho Dark Theme.
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

    # Cấu hình Dark Theme cho Matplotlib
    BG_COLOR_MPL = "#0F1113"
    TEXT_COLOR_MPL = "#E0E0E0"
    
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    ax.set_facecolor(BG_COLOR_MPL)
    fig.patch.set_facecolor(BG_COLOR_MPL)
    
    bars = ax.bar(df_plot['Tên phim'], df_plot[score_col], 
                      color=colors, edgecolor=TEXT_COLOR_MPL, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + ax.get_ylim()[1]*0.01, 
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold', rotation=45, color=TEXT_COLOR_MPL)

    # Thiết lập màu sắc và font cho biểu đồ
    ax.set_title(title, fontsize=14, color='#FF4500') # Màu nhấn Cam
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
        user_genres_list = ast.literal_eval(user_genres_str)
        if not isinstance(user_genres_list, list): user_genres_list = []
    except (ValueError, SyntaxError):
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
        score = row[score_column]
        # Xử lý Năm phát hành, đảm bảo là số nguyên
        year = int(row.get('Năm phát hành', 'N/A')) if pd.notna(row.get('Năm phát hành')) and row.get('Năm phát hành') != "" else 'N/A'
        
        # Placeholder Image URL
        placeholder_text = title.replace(' ', '+')
        # Sử dụng màu tối cho placeholder text và màu cam cho nền/chữ (phù hợp Dark Theme)
        placeholder_url = f"https://placehold.co/180x250/{index%06X}/FF4500?text={placeholder_text[:15]}..."

        
        # Dùng Score làm điểm hiển thị chính
        score_display = f"ĐIỂM: {score:.2f}" if isinstance(score, (int, float)) else "N/A"
        
        card_html = f"""
        <div class="movie-card">
            <div class="movie-poster" style="background-image: url('{placeholder_url}'); background-size: cover; background-position: center;">
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


# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    
    # Inject Dark Theme CSS
    inject_dark_theme() 
    
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username_display = "Khách" if is_guest else st.session_state['logged_in_user']
    
    st.title(f"🎬 Chào mừng, {username_display}!")
    st.sidebar.title("Menu Đề Xuất")
    
    if is_guest:
        st.header("🔥 Đề xuất Zero-Click")
        if not st.session_state['selected_intro_topics']:
            draw_interest_cards_guest()
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout, use_container_width=True): pass
            return 
        else:
            selected_topics = st.session_state['selected_intro_topics']
            selected_genre_list = []
            for topic in selected_topics:
                selected_genre_list.extend(INTRO_TOPICS.get(topic, {}).get("genres", []))
            
            st.info(f"Đang xem đề xuất cho: **{', '.join(selected_topics)}**.")
            
            if st.session_state['last_guest_result'].empty:
                zero_click_results = get_zero_click_recommendations(df_movies, selected_genre_list)
                if not zero_click_results.empty:
                    st.session_state['last_guest_result'] = zero_click_results
                    st.session_state['show_guest_plot'] = True
                else:
                    st.warning("⚠️ Không thể tạo đề xuất.")
            
            if not st.session_state['last_guest_result'].empty:
                st.subheader("✅ Phim Đề Xuất:")
                # HIỂN THỊ DƯỚI DẠNG GRID
                display_movie_grid(st.session_state['last_guest_result'], 'combined_zero_click_score')
                
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_guest_plot'], key="plot_guest_check"):
                    plot_recommendation_comparison(st.session_state['last_guest_result'], "Zero-Click")
                
                # Hiển thị DataFrame chi tiết (tùy chọn)
                with st.expander("Xem chi tiết dưới dạng bảng"):
                    st.dataframe(st.session_state['last_guest_result'], use_container_width=True)
            
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout, use_container_width=True): pass

    else:
        # --- LOGIC CHO NGƯỜI DÙNG ĐÃ ĐĂNG NHẬP ---
        df_users = st.session_state['df_users']
        username = st.session_state['logged_in_user']
        user_row = df_users[df_users['Tên người dùng'] == username]
        
        if user_row.empty:
            st.error("Lỗi: Không tìm thấy hồ sơ người dùng trong hệ thống. Vui lòng đăng nhập lại.")
            st.session_state['logged_in_user'] = None
            st.rerun()
            return
        
        # CẬP NHẬT MENU SIDEBAR THEO YÊU CẦU
        menu_choice = st.sidebar.radio(
            "Chọn chức năng:", 
            ('Đề xuất theo Tên Phim', 'Đề xuất theo AI', 'Đề xuất theo Thể loại Yêu thích', 'Đăng Xuất')
        )

        if st.sidebar.button("Đăng Xuất", on_click=logout, use_container_width=True): pass 
        st.sidebar.write("-" * 20)

        if menu_choice == 'Đề xuất theo Tên Phim':
            st.header("1️⃣ Đề xuất theo Nội dung")
            movie_titles_list = get_unique_movie_titles(df_movies)
            default_movie = st.session_state['last_sim_movie'] if st.session_state['last_sim_movie'] in movie_titles_list else movie_titles_list[0]
            
            # Đảm bảo index hợp lệ
            try:
                default_index = movie_titles_list.index(default_movie)
            except ValueError:
                default_index = 0
                default_movie = movie_titles_list[0] if movie_titles_list else ""
                
            movie_name = st.selectbox("🎥 Chọn tên phim:", options=movie_titles_list, index=default_index)
            
            weight_sim = st.slider("⚖️ Trọng số Độ giống", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("Tìm Đề Xuất", key="find_sim", type="primary"):
                result = recommend_movies_smart(movie_name, weight_sim, 1-weight_sim, df_movies, cosine_sim)
                if not result.empty:
                    st.session_state['last_sim_result'] = result
                    st.session_state['last_sim_movie'] = movie_name
                    st.session_state['show_sim_plot'] = True 
                else:
                    st.warning("Không tìm thấy đề xuất.")
                st.rerun()

            if not st.session_state['last_sim_result'].empty:
                st.subheader(f"🎬 Đề xuất cho '{st.session_state['last_sim_movie']}':")
                # HIỂN THỊ DƯỚI DẠNG GRID
                display_movie_grid(st.session_state['last_sim_result'], 'weighted_score')

                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_sim_plot'], key="plot_sim_check"):
                    plot_recommendation_comparison(st.session_state['last_sim_result'], "Tên Phim", movie_name=st.session_state['last_sim_movie'])
                
                with st.expander("Xem chi tiết dưới dạng bảng"):
                    st.dataframe(st.session_state['last_sim_result'], use_container_width=True)


        elif menu_choice == 'Đề xuất theo AI':
            st.header("2️⃣ Đề xuất theo AI (Dựa trên Hồ sơ)")
            
            is_new_registration_with_results = (
                not st.session_state['last_profile_recommendations'].empty and
                'last_profile_recommendations' in st.session_state and 
                user_row['Phim yêu thích nhất'].values[0] == "" and 
                user_row['5 phim coi gần nhất'].values[0] != "[]" 
            )

            if is_new_registration_with_results:
                    st.subheader(f"✅ Đề xuất Dành Riêng Cho Bạn (Dựa trên Thể loại đã chọn khi đăng ký):")
            
            if st.button("Tìm Đề Xuất AI", key="find_profile", type="primary"):
                recommendations = get_recommendations(username, df_movies)
                if not recommendations.empty:
                    st.session_state['last_profile_recommendations'] = recommendations
                    st.session_state['show_profile_plot'] = True 
                else:
                    st.warning("Chưa đủ dữ liệu để đề xuất.")
                st.rerun()

            if not st.session_state['last_profile_recommendations'].empty:
                recommendations = st.session_state['last_profile_recommendations']
                if not is_new_registration_with_results: 
                    st.subheader(f"✅ Đề xuất Dành Riêng Cho Bạn:")
                
                # HIỂN THỊ DƯỚI DẠNG GRID
                display_movie_grid(recommendations, 'Similarity_Score')

                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_profile_plot'], key="plot_profile_check"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI")
                
                with st.expander("Xem chi tiết dưới dạng bảng"):
                    st.dataframe(st.session_state['last_profile_recommendations'], use_container_width=True)


        elif menu_choice == 'Đề xuất theo Thể loại Yêu thích':
            st.header("3️⃣ Đề xuất theo Thể loại Yêu thích")
            
            recent_genres_str = user_row['5 phim coi gần nhất'].values[0]
            recent_genres = []
            try:
                recent_genres = ast.literal_eval(recent_genres_str)
            except:
                recent_genres = [g.strip().strip("'") for g in recent_genres_str.strip('[]').split(',') if g.strip()]
            
            if not recent_genres:
                st.warning("Bạn chưa chọn thể loại yêu thích khi đăng ký. Vui lòng đăng ký lại hoặc sử dụng chức năng khác.")
                return

            recent_genres_display = ', '.join([str(item) for item in recent_genres if str(item).strip()])

            st.info(f"Các thể loại trong hồ sơ của bạn: **{recent_genres_display}**")
            st.caption("Bấm nút bên dưới để chạy lại thuật toán đề xuất AI dựa trên các thể loại này.")

            if st.button("♻️ Chạy lại Đề xuất AI theo Thể loại này", key="rerun_profile_by_genre", type="primary"):
                recommendations = get_recommendations(username, df_movies)
                if not recommendations.empty:
                    st.session_state['last_profile_recommendations'] = recommendations
                    st.session_state['show_profile_plot'] = True 
                else:
                    st.warning("Chưa đủ dữ liệu để đề xuất.")
                st.rerun()
            
            # Hiển thị kết quả đề xuất gần nhất nếu có
            if not st.session_state['last_profile_recommendations'].empty:
                st.write("---")
                st.subheader("Kết quả Đề xuất AI gần nhất:")
                # HIỂN THỊ DƯỚI DẠNG GRID
                display_movie_grid(st.session_state['last_profile_recommendations'], 'Similarity_Score')

                if st.checkbox("📊 Hiển thị Biểu đồ", key="plot_profile_check_genre"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI (Theo Thể loại)")
                
                with st.expander("Xem chi tiết dưới dạng bảng"):
                    st.dataframe(st.session_state['last_profile_recommendations'], use_container_width=True)


        elif menu_choice == 'Đăng Xuất':
            logout()


if __name__ == '__main__':
    # Streamlit Config
    st.set_page_config(
        page_title="Movie Recommender AI", 
        page_icon="🎬", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    df_movies, cosine_sim = load_and_preprocess_static_data()
    initialize_user_data()
    
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies, cosine_sim)
