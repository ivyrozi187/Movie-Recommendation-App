import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random 
import matplotlib.colors as mcolors

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" 

# --- CẤU HÌNH DANH SÁCH THỂ LOẠI (TOPICS) THEO YÊU CẦU ---
# Danh sách màu sắc Pastel (Pastel Dream Palette) cho các thẻ
COLOR_PALETTE = [
    ("#FFC3A0", "#FFAD7F", "#E69C7A"), # Peach
    ("#35D0BA", "#45B8AC", "#30A89C"), # Mint Green
    ("#F8B195", "#F67280", "#E87A90"), # Salmon Pink
    ("#E6A4B4", "#F4C4D4", "#D899A9"), # Baby Pink
    ("#6C5B7B", "#C06C84", "#A85C74"), # Muted Violet
    ("#84B9A7", "#A4C3A3", "#90B090"), # Sage Green
    ("#E9F2F9", "#A2C3CC", "#8BB0BC"), # Light Blue
    ("#B39EB5", "#D2B4DE", "#A18EC8"), # Lavender
    ("#87CEEB", "#ADD8E6", "#73B8D4"), # Sky Blue
    ("#F0E68C", "#FFFACD", "#D8D07C"), # Khaki Yellow
    ("#D2D792", "#E0E3B6", "#C1C585"), # Muted Lime
    ("#FFDAB9", "#FFE4C4", "#E6C9A9"), # Peach Puff
]

# Danh sách 23 thể loại từ dữ liệu
GENRES_VI = [
    "Phim Hành Động", "Phim Giả Tượng", "Phim Hài", "Phim Kinh Dị", 
    "Phim Phiêu Lưu", "Phim Chính Kịch", "Phim Khoa Học Viễn Tưởng", 
    "Phim Gây Thú Vị", "Phim Gia Đình", "Phim Hoạt Hình", "Phim Lãng Mạn", 
    "Phim Tài Liệu", "Phim Chiến Tranh", "Phim Bí Ẩn", "Phim Hình Sự", 
    "Phim Viễn Tây", "Phim Cổ Trang", "Phim Nhạc", "Phim Lịch Sử", 
    "Phim Thần Thoại", "Phim Truyền Hình", "Chương Trình TV", "Phim Khác"
]

# Tạo dictionary ánh xạ tự động
INTRO_TOPICS = {}
for i, genre in enumerate(GENRES_VI):
    color, gradient, hover_color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    INTRO_TOPICS[genre] = {
        "genres": [genre],
        "color": color, 
        "gradient": gradient,
        "hover_color": hover_color
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

# Biến trạng thái mới cho Đăng ký (TOPICS)
if 'selected_reg_topics' not in st.session_state: st.session_state['selected_reg_topics'] = set()


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
            df_movies['Năm phát hành'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(pd.Timestamp('now').year)
            max_year = df_movies['Năm phát hành'].max()
            min_year = df_movies['Năm phát hành'].min()
            if max_year > min_year:
                 df_movies['recency_score'] = (df_movies['Năm phát hành'] - min_year) / (max_year - min_year)
            else:
                 df_movies['recency_score'] = 0.5 
        else:
            df_movies['recency_score'] = df_movies["popularity_norm"] * 0.1 

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
            df_users = load_data(USER_DATA_FILE)
            df_users.columns = [col.strip() for col in df_users.columns]
            
            for col in REQUIRED_USER_COLUMNS:
                if col not in df_users.columns:
                    df_users[col] = "" 
            
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
    st.session_state['selected_reg_topics'] = set()
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
# UI: CÁC HÀM VẼ GIAO DIỆN VÀ CSS (PASTEL THEME)
# ------------------------------------------------------------------------------

def inject_pastel_theme():
    """Tiêm CSS để tạo giao diện Pastel Theme cho Streamlit."""
    # Màu sắc chủ đạo Pastel
    BG_COLOR = "#FFF7F7"      # Nền rất nhạt (Creamy White/Very Light Pink)
    TEXT_COLOR = "#333333"    # Màu chữ đậm
    PRIMARY_COLOR = "#FFAD7F" # Màu cam đào (Peach) - Dùng cho nút chính
    SECONDARY_BG = "#EAE7DC"  # Sidebar và background phụ (Grayish Beige)
    ACCENT_COLOR = "#C06C84"  # Màu nhấn (Muted Rose) - Cho tiêu đề/biểu đồ
    
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
            border-right: 2px solid {ACCENT_COLOR}30;
        }}
        
        /* Header và Title */
        h1, h2, h3, h4 {{
            color: {ACCENT_COLOR};
            font-weight: 700;
            font-family: 'Georgia', serif; /* Tạo cảm giác sang trọng hơn */
        }}
        
        /* Nút chính (Đăng ký/Tìm kiếm) */
        .stButton button {{
            border-radius: 8px;
            padding: 10px 15px;
            font-weight: bold;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }}
        
        /* Nút Primary (ví dụ: nút "Hoàn Tất Đăng Ký") */
        .stButton button[kind="primary"] {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: 2px solid {PRIMARY_COLOR};
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        .stButton button[kind="primary"]:hover {{
            background-color: {ACCENT_COLOR};
            border-color: {ACCENT_COLOR};
            transform: translateY(-2px);
        }}

        /* Info boxes */
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"] {{
            background-color: #F8F0E3; /* Nền giấy */
            border-left: 5px solid {PRIMARY_COLOR};
            border-radius: 8px;
            padding: 10px;
            color: {TEXT_COLOR};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
    </style>
    """, unsafe_allow_html=True)


def draw_registration_topic_cards():
    """Vẽ giao diện chọn chủ đề (Topic) thay vì chọn từng genre lẻ (Pastel Card Design)."""
    
    st.markdown("### 💖 Hãy chọn Thể loại Yêu thích của bạn!")
    st.caption("Chọn ít nhất 3 thể loại để cá nhân hóa hồ sơ của bạn:")

    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(4) 
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        is_selected = topic in st.session_state['selected_reg_topics']
        
        # Style động: Viền sáng và hiệu ứng nổi bật khi được chọn
        border_style = "border: 3px solid #C06C84;" if is_selected else "border: 1px solid #C06C8450;"
        selected_shadow = "box-shadow: 0 0 15px rgba(255, 173, 127, 0.9);" if is_selected else "box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);"
        
        # Gradient nền
        bg_gradient = f"linear-gradient(145deg, {data['color']}AA, {data['gradient']}AA)"

        # Tạo style riêng cho từng nút
        btn_style = f"""
            background: {bg_gradient};
            color: #333333; /* Chữ đậm */
            border-radius: 12px; /* Bo góc nhiều hơn */
            height: 90px; 
            font-weight: 700;
            font-size: 1.0rem;
            width: 100%;
            margin-bottom: 12px;
            
            {border_style}
            {selected_shadow}
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); /* Transition mượt hơn */
        """
        
        # --- HIỆU ỨNG HOVER NỔI BẬT (3D LIFT) ---
        hover_style = f"""
            div[data-testid="stButton"] button[key="reg_topic_{topic}"]:hover {{
                background: linear-gradient(145deg, {data['hover_color']}, {data['gradient']});
                transform: translateY(-5px); /* Nhấc lên */
                box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3); /* Shadow lớn hơn */
                border-color: {data['hover_color']};
                color: white; /* Đổi màu chữ khi hover */
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
            
            # Inject CSS chi tiết vào nút vừa tạo
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
                        filter: brightness(95%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: {data['hover_color']};
                    }}
                </style>
            """, unsafe_allow_html=True)


def draw_interest_cards_guest():
    """Giao diện thẻ cho chế độ Khách (Guest) - Chỉ chọn 1."""
    st.header("✨ Bạn đang quan tâm gì?")
    st.markdown("Chọn một chủ đề để nhận đề xuất theo xu hướng toàn cầu:")
    
    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(4)
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        bg_gradient = f"linear-gradient(135deg, {data['color']}, {data['gradient']})"
        
        btn_style = f"""
            background: {bg_gradient};
            color: white;
            border-radius: 12px;
            height: 100px;
            font-weight: 700;
            font-size: 1.0rem;
            width: 100%;
            margin-bottom: 12px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
            border: none;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        """
        
        hover_style = f"""
            div[data-testid="stButton"] button[key="guest_{topic}"]:hover {{
                background: {data['hover_color']};
                transform: scale(1.05) rotate(1deg); /* Hiệu ứng xoay nhẹ và phóng to */
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
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
                    div[data-testid="stButton"] button[key="guest_{topic}"]:active {{
                        transform: scale(0.98);
                        filter: brightness(90%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)


# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ
# ==============================================================================

# Tạo danh sách màu sắc rực rỡ và dễ phân biệt
def get_vibrant_colors(n):
    """Tạo n màu sắc Pastel/Muted (dịu) để phù hợp với theme."""
    cmap = plt.cm.get_cmap('Pastel1', n) # Đổi sang Pastel1
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]
    # Thêm màu nhấn Pastel đậm hơn
    colors[0] = '#FFAD7F'
    colors[1] = '#C06C84'
    return colors

def plot_recommendation_comparison(df_results, recommendation_type, movie_name=None):
    """Vẽ biểu đồ so sánh điểm số đề xuất (hoặc độ phổ biến) của các phim."""
    if df_results.empty:
        st.warning("Không có dữ liệu để vẽ biểu đồ.")
        return

    # 1. Xác định Cột điểm và Tiêu đề
    if 'weighted_score' in df_results.columns:
        score_col = 'weighted_score'
        y_label = "Điểm Đề xuất Tổng hợp (Similarity + Popularity)"
        title_prefix = f"So sánh Đề xuất theo Tên Phim ('{movie_name}')"
    elif 'Similarity_Score' in df_results.columns:
        score_col = 'Similarity_Score'
        y_label = "Điểm Giống nhau (Genre Match)"
        title_prefix = f"So sánh Đề xuất theo AI (Genre Score)"
    elif 'combined_zero_click_score' in df_results.columns:
        score_col = 'combined_zero_click_score'
        y_label = "Điểm Zero-Click (Global Trend + Genre Boost)"
        title_prefix = "So sánh Đề xuất Zero-Click"
    else:
        score_col = 'Độ phổ biến'
        y_label = "Độ Phổ Biến"
        title_prefix = "So sánh Độ Phổ Biến"
        
    title = f"{title_prefix}\n({recommendation_type})"

    # Sắp xếp theo điểm số để biểu đồ trực quan hơn
    df_plot = df_results.sort_values(by=score_col, ascending=True).copy()
    
    # 2. Tạo màu sắc riêng cho mỗi phim (Pastel)
    num_movies = len(df_plot)
    colors = get_vibrant_colors(num_movies)

    # 3. Vẽ biểu đồ CỘT DỌC
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # Dùng biểu đồ cột dọc
    bars = ax.bar(df_plot['Tên phim'], df_plot[score_col], 
                   color=colors, edgecolor='#333333', alpha=0.9)

    # 4. Thêm nhãn giá trị lên thanh
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + ax.get_ylim()[1]*0.01, 
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold', color='#333333', rotation=45)

    # Thiết lập màu sắc và font cho biểu đồ
    ax.set_title(title, fontsize=16, color='#C06C84', weight='bold')
    ax.set_xlabel("Tên Phim", color='#333333')
    ax.set_ylabel(y_label, color='#333333')
    ax.tick_params(axis='x', colors='#333333')
    ax.tick_params(axis='y', colors='#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.set_facecolor('#FFF7F7')
    
    # Xoay nhãn trục X để tránh chồng chéo
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Điều chỉnh giới hạn trục Y
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#C06C8450')
    
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
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'combined_zero_click_score']].head(num_recommendations)


def get_recommendations(username, df_movies, num_recommendations=10):
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    user_genres_str = user_row['5 phim coi gần nhất'].iloc[0]
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
    
    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    if favorite_movie:
        favorite_movie_genres = df_movies[df_movies['Tên phim'] == favorite_movie]['parsed_genres'].iloc[0] if not df_movies[df_movies['Tên phim'] == favorite_movie].empty else set()
        user_genres.update(favorite_movie_genres)

    if not user_genres: return pd.DataFrame()

    candidate_movies = df_movies[df_movies['Tên phim'] != favorite_movie].copy()
    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(lambda x: len(x.intersection(user_genres)))

    recommended_df = candidate_movies.sort_values(by=['Similarity_Score', 'Độ phổ biến'], ascending=[False, False])
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)

def get_movie_index(movie_name, df_movies):
    try:
        # Tìm kiếm không phân biệt chữ hoa, chữ thường và xóa khoảng trắng
        idx = df_movies[df_movies['Tên phim'].str.lower().str.strip() == movie_name.lower().strip()].index[0]
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
    df_result = df_result.drop(df_result[df_result['Tên phim'].str.lower().str.strip() == movie_name.lower().strip()].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)
    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)


# ==============================================================================
# IV. TRANG XÁC THỰC (LOGIN / REGISTER)
# ==============================================================================

def register_user(username, selected_topics):
    df_users = st.session_state['df_users']
    
    if username in df_users['Tên người dùng'].values:
        st.error("Tên người dùng đã tồn tại!")
        return False
    
    if not username:
        st.error("Tên người dùng không được để trống!")
        return False
    
    # 1. Tạo danh sách Genres từ Topics đã chọn
    genres_list = []
    for topic in selected_topics:
        genres_list.extend(INTRO_TOPICS.get(topic, {}).get("genres", []))
    
    # 2. Xử lý ID mới
    new_id = df_users['ID'].max() + 1 if not df_users.empty and df_users['ID'].max() else 1
    
    # 3. Tạo record mới
    new_user = pd.DataFrame({
        'ID': [new_id],
        'Tên người dùng': [username],
        # Lưu genres đã chọn vào cột '5 phim coi gần nhất' (dùng để khởi tạo profile)
        '5 phim coi gần nhất': [repr(genres_list)], 
        'Phim yêu thích nhất': [""]
    })
    
    st.session_state['df_users'] = pd.concat([df_users, new_user], ignore_index=True)
    st.session_state['logged_in_user'] = username
    st.session_state['selected_reg_topics'] = set()
    
    # Tự động chạy đề xuất AI lần đầu để hiển thị ngay
    df_movies = load_and_preprocess_static_data()[0]
    initial_recommendations = get_recommendations(username, df_movies)
    st.session_state['last_profile_recommendations'] = initial_recommendations
    st.success(f"Đăng ký thành công! Chào mừng {username}. Đang tạo đề xuất ban đầu...")
    st.rerun()

def authentication_page(df_movies, cosine_sim):
    inject_pastel_theme()
    
    st.title("🍿 DreamStream: Đề xuất Phim Cá nhân")
    st.subheader("Bắt đầu trải nghiệm xem phim của bạn!")
    
    # Tabs cho Login và Register
    login_tab, register_tab, guest_tab = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    # --- TAB ĐĂNG NHẬP ---
    with login_tab:
        st.markdown("#### 🔑 Đăng Nhập Tài Khoản")
        login_username = st.text_input("Tên người dùng:", key="login_user")
        
        if st.button("Đăng Nhập", key="login_btn", type="primary"):
            df_users = st.session_state['df_users']
            if login_username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = login_username
                st.success(f"Chào mừng trở lại, {login_username}!")
                st.rerun()
            else:
                st.error("Tên người dùng không tồn tại. Vui lòng thử lại hoặc đăng ký.")

    # --- TAB ĐĂNG KÝ ---
    with register_tab:
        st.markdown("#### 📝 Đăng Ký Tài Khoản Mới")
        reg_username = st.text_input("Tên người dùng bạn muốn tạo:", key="reg_user")
        
        st.write("---")
        draw_registration_topic_cards()
        st.write("---")
        
        if st.button("Hoàn Tất Đăng Ký", key="reg_btn", type="primary", use_container_width=True):
            if len(st.session_state['selected_reg_topics']) < 3:
                st.warning("Vui lòng chọn ít nhất 3 thể loại yêu thích.")
            else:
                register_user(reg_username, st.session_state['selected_reg_topics'])

    # --- TAB CHẾ ĐỘ KHÁCH ---
    with guest_tab:
        st.markdown("#### 🚶 Chế Độ Khách (Zero-Click)")
        st.info("Trải nghiệm hệ thống đề xuất ngay lập tức mà không cần đăng ký. Bạn sẽ nhận được các phim theo xu hướng toàn cầu và sở thích tạm thời của bạn.")
        if st.button("Truy Cập với tư cách Khách", key="guest_btn", use_container_width=True):
            login_as_guest()


# ==============================================================================
# V. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    
    # Inject Pastel Theme CSS
    inject_pastel_theme() 
    
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username_display = "Khách" if is_guest else st.session_state['logged_in_user']
    
    st.title(f"🎬 Chào mừng, {username_display}!")
    st.sidebar.title("Menu Chức Năng")
    
    if is_guest:
        # LOGIC CHO GUEST MODE
        st.header("🔥 Đề xuất Zero-Click (Theo Xu hướng)")
        
        if not st.session_state['selected_intro_topics']:
            draw_interest_cards_guest()
            st.sidebar.write("---")
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout, use_container_width=True): pass
            return 
        else:
            selected_topics = st.session_state['selected_intro_topics']
            selected_genre_list = []
            for topic in selected_topics:
                selected_genre_list.extend(INTRO_TOPICS.get(topic, {}).get("genres", []))
            
            st.info(f"Đang xem đề xuất cho: **{', '.join(selected_topics)}**. Dữ liệu được làm mới sau mỗi lần chọn.")
            
            if st.session_state['last_guest_result'].empty:
                zero_click_results = get_zero_click_recommendations(df_movies, selected_genre_list)
                if not zero_click_results.empty:
                    st.session_state['last_guest_result'] = zero_click_results
                    st.session_state['show_guest_plot'] = True
                else:
                    st.warning("⚠️ Không thể tạo đề xuất.")
            
            if not st.session_state['last_guest_result'].empty:
                st.subheader("✅ 15 Phim Đề Xuất Phù Hợp:")
                st.dataframe(st.session_state['last_guest_result'].style.background_gradient(cmap='YlGnBu'), use_container_width=True)
                
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_guest_plot'], key="plot_guest_check"):
                    plot_recommendation_comparison(st.session_state['last_guest_result'], "Zero-Click")
            
            st.sidebar.write("---")
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout, use_container_width=True): pass

    else:
        # --- LOGIC CHO NGƯỜI DÙNG ĐÃ ĐĂNG NHẬP ---
        df_users = st.session_state['df_users']
        username = st.session_state['logged_in_user']
        user_row = df_users[df_users['Tên người dùng'] == username]
        
        if user_row.empty:
            st.error("Lỗi: Không tìm thấy hồ sơ người dùng. Vui lòng đăng nhập lại.")
            st.session_state['logged_in_user'] = None
            st.rerun()
            return
        
        menu_choice = st.sidebar.radio(
            "Chọn chức năng:", 
            ('Đề xuất theo Tên Phim', 'Đề xuất theo AI', 'Đề xuất theo Thể loại Yêu thích', 'Đăng Xuất'),
            index=0
        )

        st.sidebar.write("---")
        if st.sidebar.button("Đăng Xuất", on_click=logout, use_container_width=True): pass 
        st.sidebar.write("---")

        if menu_choice == 'Đề xuất theo Tên Phim':
            st.header("1️⃣ Đề xuất theo Nội dung (Content-Based)")
            st.info("Tìm kiếm các phim có cùng đạo diễn, diễn viên và thể loại với phim bạn chọn.")
            
            movie_titles_list = get_unique_movie_titles(df_movies)
            default_movie = st.session_state['last_sim_movie'] if st.session_state['last_sim_movie'] in movie_titles_list else movie_titles_list[0]
            movie_name = st.selectbox("🎥 Chọn tên phim:", options=movie_titles_list, index=movie_titles_list.index(default_movie) if default_movie in movie_titles_list else 0)
            
            weight_sim = st.slider("⚖️ Trọng số Độ giống (Càng cao càng giống nhau về nội dung)", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("Tìm Đề Xuất", key="find_sim", type="primary", use_container_width=True):
                result = recommend_movies_smart(movie_name, weight_sim, 1-weight_sim, df_movies, cosine_sim)
                if not result.empty:
                    st.session_state['last_sim_result'] = result
                    st.session_state['last_sim_movie'] = movie_name
                    st.session_state['show_sim_plot'] = True 
                else:
                    st.warning("Không tìm thấy đề xuất cho phim này.")
                st.rerun()

            if not st.session_state['last_sim_result'].empty:
                st.subheader(f"🎬 Đề xuất cho '{st.session_state['last_sim_movie']}':")
                st.dataframe(st.session_state['last_sim_result'].style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_sim_plot'], key="plot_sim_check"):
                    plot_recommendation_comparison(st.session_state['last_sim_result'], "Tên Phim", movie_name=st.session_state['last_sim_movie'])

        elif menu_choice == 'Đề xuất theo AI':
            st.header("2️⃣ Đề xuất theo AI (Profile-Based)")
            
            is_new_registration_with_results = (
                not st.session_state['last_profile_recommendations'].empty and
                'last_profile_recommendations' in st.session_state and 
                user_row['Phim yêu thích nhất'].iloc[0] == "" and 
                user_row['5 phim coi gần nhất'].iloc[0] != "[]" 
            )

            if is_new_registration_with_results:
                st.success(f"Dữ liệu hồ sơ của bạn đã được khởi tạo thành công. Đề xuất ban đầu:")
                st.info("Các đề xuất này dựa trên các thể loại bạn đã chọn khi đăng ký.")
            
            if st.button("Tìm Đề Xuất AI", key="find_profile", type="primary", disabled=is_new_registration_with_results, use_container_width=True):
                recommendations = get_recommendations(username, df_movies)
                if not recommendations.empty:
                    st.session_state['last_profile_recommendations'] = recommendations
                    st.session_state['show_profile_plot'] = True 
                else:
                    st.warning("Chưa đủ dữ liệu (phim đã xem/thích) để cá nhân hóa đề xuất.")
                st.rerun()

            if not st.session_state['last_profile_recommendations'].empty:
                st.subheader(f"✅ Đề xuất Dành Riêng Cho Bạn:")
                
                recommendations = st.session_state['last_profile_recommendations']
                st.dataframe(recommendations.style.background_gradient(cmap='YlGn'), use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_profile_plot'], key="plot_profile_check"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI")

        elif menu_choice == 'Đề xuất theo Thể loại Yêu thích':
            st.header("3️⃣ Đề xuất theo Thể loại Yêu thích")
            st.info("Xem lại các thể loại đã sử dụng để tạo hồ sơ ban đầu của bạn và chạy lại thuật toán.")
            
            recent_genres_str = user_row['5 phim coi gần nhất'].iloc[0]
            recent_genres = []
            try:
                # Cố gắng chuyển đổi chuỗi genres (được lưu bằng repr()) thành list
                recent_genres = ast.literal_eval(recent_genres_str)
            except:
                recent_genres = [g.strip().strip("'") for g in recent_genres_str.strip('[]').split(',') if g.strip()]
            
            recent_genres_display = ', '.join([f"**{str(item)}**" for item in recent_genres if str(item).strip()])

            if recent_genres_display:
                st.markdown(f"Các thể loại trong hồ sơ của bạn: {recent_genres_display}")
            else:
                st.warning("Hồ sơ của bạn chưa có thông tin thể loại yêu thích. Vui lòng đăng ký lại để thêm hoặc sử dụng chức năng Đề xuất theo Tên Phim.")
                return

            if st.button("♻️ Chạy lại Đề xuất AI theo Thể loại này", key="rerun_profile_by_genre", type="primary", use_container_width=True):
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
                st.dataframe(st.session_state['last_profile_recommendations'].style.background_gradient(cmap='YlGn'), use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", key="plot_profile_check_genre"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI (Theo Thể loại)")
        elif menu_choice == 'Đăng Xuất':
            st.header("Tạm biệt! 👋")
            st.info("Cảm ơn bạn đã sử dụng DreamStream. Vui lòng nhấn nút Đăng Xuất ở Sidebar để thoát.")


if __name__ == '__main__':
    df_movies, cosine_sim = load_and_preprocess_static_data()
    initialize_user_data()
    
    # Đặt cấu hình trang
    st.set_page_config(
        page_title="DreamStream - Đề xuất Phim",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    if df_movies.empty or cosine_sim.size == 0:
        st.stop() # Dừng nếu dữ liệu lỗi
    
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies, cosine_sim)
