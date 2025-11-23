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

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" 

# --- CẤU HÌNH DANH SÁCH THỂ LOẠI (TOPICS) THEO YÊU CẦU ---
# Danh sách màu sắc (Palette hiện đại) để luân phiên cho các thẻ
COLOR_PALETTE = [
    ("#e11d48", "#fb7185"), # Rose
    ("#ea580c", "#fb923c"), # Orange
    ("#d97706", "#fbbf24"), # Amber
    ("#65a30d", "#a3e635"), # Lime
    ("#059669", "#34d399"), # Emerald
    ("#0891b2", "#22d3ee"), # Cyan
    ("#2563eb", "#60a5fa"), # Blue
    ("#4f46e5", "#818cf8"), # Indigo
    ("#7c3aed", "#a78bfa"), # Violet
    ("#c026d3", "#e879f9"), # Fuchsia
    ("#be123c", "#fda4af"), # Pink
    ("#475569", "#94a3b8"), # Slate
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
    color, gradient = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    INTRO_TOPICS[genre] = {
        "genres": [genre], # Ánh xạ trực tiếp 1-1
        "color": color, 
        "gradient": gradient
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
            
            # --- FIX CHO LỖI KEYERROR: Đảm bảo các cột cần thiết tồn tại ---
            for col in REQUIRED_USER_COLUMNS:
                if col not in df_users.columns:
                    # Thêm cột bị thiếu với giá trị mặc định là chuỗi rỗng
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
# UI: VẼ THẺ CHỦ ĐỀ CHO ĐĂNG KÝ (Thay thế phần chọn genre cũ)
# ------------------------------------------------------------------------------
def draw_registration_topic_cards():
    """Vẽ giao diện chọn chủ đề (Topic) thay vì chọn từng genre lẻ."""
    
    st.markdown("### Bạn thích thể loại nào?")
    st.caption("Chọn các thể loại bạn thích để chúng tôi xây dựng hồ sơ ban đầu:")

    # CSS chung cho nút Streamlit, đặc biệt là nút trong cột
    st.markdown("""
    <style>
        /* Đảm bảo nút trong giao diện chọn thể loại có nền gradient và không bị ảnh hưởng bởi style Streamlit mặc định */
        div[data-testid*="stButton"] > button {
             border: none; 
             /* Đặt transition cho các hiệu ứng CSS */
             transition: all 0.2s ease-in-out;
        }
    </style>
    """, unsafe_allow_html=True)

    topics = list(INTRO_TOPICS.keys())
    # Tăng số cột lên 4 để chứa nhiều thể loại hơn cho gọn
    cols = st.columns(4) 
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        is_selected = topic in st.session_state['selected_reg_topics']
        
        # Style động: Nếu chọn thì có viền sáng/shadow
        # Thay đổi box-shadow để trông tinh tế hơn khi được chọn
        border_style = "border: 3px solid #f63366;" if is_selected else "border: none;"
        selected_shadow = "box-shadow: 0 0 18px rgba(246, 51, 102, 0.7);" if is_selected else "box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);"
        opacity = "1.0" if is_selected else "0.9"
        
        # Tạo style riêng cho từng nút
        btn_style = f"""
            /* Base style - sử dụng gradient */
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 10px;
            height: 80px; 
            font-weight: bold;
            font-size: 0.95rem;
            width: 100%;
            margin-bottom: 8px;
            
            {border_style}
            {selected_shadow}
            opacity: {opacity};
            cursor: pointer;
            
            /* Dùng flexbox để căn giữa chữ */
            display: flex; 
            align-items: center; 
            justify-content: center;
            transition: all 0.2s ease-in-out;
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
                    /* Style cơ bản (áp dụng cho cả trạng thái đã chọn) */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"] {{
                        {btn_style}
                    }}
                    /* Hiệu ứng HOVER: Sáng hơn (115%), nâng nhẹ (1.03), bóng sâu hơn */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"]:hover {{
                        filter: brightness(115%);
                        transform: scale(1.03);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
                        border-color: #f63366 !important; /* Luôn có viền màu nổi bật khi hover */
                        opacity: 1.0;
                        color: white;
                    }}
                    /* Hiệu ứng ACTIVE/CLICK: nhấn chìm */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"]:active {{
                        transform: scale(0.98);
                        filter: brightness(90%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)


def register_new_user_form(df_movies, cosine_sim):
    """
    Form đăng ký người dùng mới. 
    Đã CẬP NHẬT: Sau khi đăng ký thành công, sẽ TỰ ĐỘNG ĐỀ XUẤT PHIM.
    """
    st.header("📝 Đăng Ký Tài Khoản Mới")
    st.info("📢 Người dùng mới sẽ chỉ tồn tại trong phiên làm việc hiện tại.")

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
        # Gọi hàm đề xuất cho người dùng mới
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
        submitted = st.form_submit_button("Đăng Nhập")
        
        if submitted:
            if username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Đăng nhập thành công! Chào mừng, {username}.")
                st.rerun() 
            else:
                st.error("❌ Tên người dùng không tồn tại.")

def authentication_page(df_movies, cosine_sim):
    """Trang Xác thực."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',), use_container_width=True)
    with col2:
        st.button("Đăng Ký", key="btn_register", on_click=set_auth_mode, args=('register',), use_container_width=True)

    st.write("---")
    
    if st.session_state['auth_mode'] == 'login':
        login_form()
        st.write("")
        st.subheader("Hoặc:")
        st.button("🚀 Thử Dùng Với Chế Độ Khách (Zero-Click)", key="btn_guest", on_click=login_as_guest)
    
    elif st.session_state['auth_mode'] == 'register':
        # Truyền thêm cosine_sim vào đây để có thể gọi hàm get_recommendations bên trong
        register_new_user_form(df_movies, cosine_sim)

# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ
# ==============================================================================

# Tạo danh sách màu sắc rực rỡ và dễ phân biệt
def get_vibrant_colors(n):
    """Tạo n màu sắc rực rỡ và dễ phân biệt."""
    # Dùng colormap 'hsv' để lấy các màu phân bổ đều trên bánh xe màu
    hsv_map = plt.cm.get_cmap('hsv', n)
    # Chuyển đổi từ RGB sang mã HEX
    colors = [mcolors.rgb2hex(hsv_map(i)[:3]) for i in range(n)]
    return colors

def plot_recommendation_comparison(df_results, recommendation_type, movie_name=None):
    """
    Vẽ biểu đồ so sánh điểm số đề xuất (hoặc độ phổ biến) của các phim.
    Mỗi phim một màu riêng biệt. (Đã chuyển sang cột dọc)
    """
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
        # Fallback nếu không tìm thấy cột điểm, dùng Độ phổ biến
        score_col = 'Độ phổ biến'
        y_label = "Độ Phổ Biến"
        title_prefix = "So sánh Độ Phổ Biến"
        
    title = f"{title_prefix}\n({recommendation_type})"

    # Sắp xếp theo điểm số để biểu đồ trực quan hơn
    df_plot = df_results.sort_values(by=score_col, ascending=True).copy()
    
    # 2. Tạo màu sắc riêng cho mỗi phim
    num_movies = len(df_plot)
    colors = get_vibrant_colors(num_movies)

    # 3. Vẽ biểu đồ CỘT DỌC
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # Dùng biểu đồ cột dọc
    bars = ax.bar(df_plot['Tên phim'], df_plot[score_col], 
                   color=colors, edgecolor='black', alpha=0.8)

    # 4. Thêm nhãn giá trị lên thanh
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + ax.get_ylim()[1]*0.01, 
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold', rotation=45)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Tên Phim") # Trục X là Tên Phim
    ax.set_ylabel(y_label) # Trục Y là Điểm
    
    # Xoay nhãn trục X để tránh chồng chéo
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Điều chỉnh giới hạn trục Y
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
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'combined_zero_click_score']].head(num_recommendations)


def get_recommendations(username, df_movies, num_recommendations=10):
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame() # Kiểm tra rỗng

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
    
    # Lấy phim yêu thích (nếu có) để boost thêm
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
    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)


# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def draw_interest_cards_guest():
    """Giao diện thẻ cho chế độ Khách (Guest) - Chỉ chọn 1. Đã áp dụng CSS mới."""
    st.header("Bạn đang quan tâm gì?")
    st.markdown("Chọn một chủ đề để nhận đề xuất ngay:")
    
    st.markdown("""
    <style>
        /* Đặt style chung cho tất cả các nút card */
        div[data-testid*="stButton"] button {
            border: none;
            transition: all 0.2s ease-in-out;
        }
    </style>
    """, unsafe_allow_html=True)

    topics = list(INTRO_TOPICS.keys())
    # Tăng số cột lên 4 cho gọn
    cols = st.columns(4)
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        btn_style = f"""
            /* Base style - sử dụng gradient */
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 10px;
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
        with cols[i % 4]:
            st.button(topic, key=f"guest_{topic}", on_click=select_topic, args=(topic,), use_container_width=True)
            st.markdown(f"""
                <style>
                    div[data-testid="stButton"] button[key="guest_{topic}"] {{ 
                        {btn_style} 
                    }}
                    /* Hiệu ứng HOVER: Sáng hơn (115%), nâng nhẹ (1.03), bóng sâu hơn */
                    div[data-testid="stButton"] button[key="guest_{topic}"]:hover {{
                        filter: brightness(115%);
                        transform: scale(1.03);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
                        color: white;
                    }}
                    /* Hiệu ứng ACTIVE/CLICK: nhấn chìm */
                    div[data-testid="stButton"] button[key="guest_{topic}"]:active {{
                        transform: scale(0.98);
                        filter: brightness(90%);
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)

def main_page(df_movies, cosine_sim):
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username_display = "Khách" if is_guest else st.session_state['logged_in_user']
    
    st.title(f"🎬 Chào mừng, {username_display}!")
    st.sidebar.title("Menu Đề Xuất")
    
    if is_guest:
        # Giữ nguyên logic Guest Mode
        st.header("🔥 Đề xuất Zero-Click")
        if not st.session_state['selected_intro_topics']:
            draw_interest_cards_guest()
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout): pass
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
                st.subheader("✅ 15 Phim Đề Xuất:")
                st.dataframe(st.session_state['last_guest_result'], use_container_width=True)
                
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_guest_plot'], key="plot_guest_check"):
                    plot_recommendation_comparison(st.session_state['last_guest_result'], "Zero-Click")
            
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout): pass

    else:
        # --- LOGIC CHO NGƯỜI DÙNG ĐÃ ĐĂNG NHẬP ---
        df_users = st.session_state['df_users']
        username = st.session_state['logged_in_user']
        user_row = df_users[df_users['Tên người dùng'] == username]
        
        # Kiểm tra nếu user_row rỗng (có thể do lỗi tải data hoặc user mới bị mất)
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

        if st.sidebar.button("Đăng Xuất", on_click=logout): pass 
        st.sidebar.write("-" * 20)

        if menu_choice == 'Đề xuất theo Tên Phim':
            # Giữ nguyên logic Content-Based
            st.header("1️⃣ Đề xuất theo Nội dung")
            movie_titles_list = get_unique_movie_titles(df_movies)
            default_movie = st.session_state['last_sim_movie'] if st.session_state['last_sim_movie'] in movie_titles_list else movie_titles_list[0]
            movie_name = st.selectbox("🎥 Chọn tên phim:", options=movie_titles_list, index=movie_titles_list.index(default_movie))
            
            weight_sim = st.slider("⚖️ Trọng số Độ giống", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("Tìm Đề Xuất", key="find_sim"):
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
                st.dataframe(st.session_state['last_sim_result'], use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_sim_plot'], key="plot_sim_check"):
                    plot_recommendation_comparison(st.session_state['last_sim_result'], "Tên Phim", movie_name=st.session_state['last_sim_movie'])

        elif menu_choice == 'Đề xuất theo AI':
            # CẬP NHẬT TIÊU ĐỀ
            st.header("2️⃣ Đề xuất theo AI")
            
            # Logic TỰ ĐỘNG GỌI ĐỀ XUẤT NẾU LÀ ĐĂNG KÝ MỚI
            is_new_registration_with_results = (
                not st.session_state['last_profile_recommendations'].empty and
                'last_profile_recommendations' in st.session_state and 
                user_row['Phim yêu thích nhất'].iloc[0] == "" and 
                user_row['5 phim coi gần nhất'].iloc[0] != "[]" 
            )

            if is_new_registration_with_results:
                 st.subheader(f"✅ Đề xuất Dành Riêng Cho Bạn (Dựa trên Thể loại đã chọn khi đăng ký):")
            elif st.button("Tìm Đề Xuất AI", key="find_profile"):
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
                
                st.dataframe(recommendations, use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_profile_plot'], key="plot_profile_check"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI")

        elif menu_choice == 'Đề xuất theo Thể loại Yêu thích':
            # --- LOGIC MỚI: HIỂN THỊ THỂ LOẠI VÀ CHẠY LẠI ĐỀ XUẤT ---
            st.header("3️⃣ Đề xuất theo Thể loại Yêu thích")
            
            # Lấy dữ liệu an toàn
            recent_genres_str = user_row['5 phim coi gần nhất'].iloc[0]
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

            if st.button("♻️ Chạy lại Đề xuất AI theo Thể loại này", key="rerun_profile_by_genre"):
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
                st.dataframe(st.session_state['last_profile_recommendations'], use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", key="plot_profile_check_genre"):
                    plot_recommendation_comparison(st.session_state['last_profile_recommendations'], "AI (Theo Thể loại)")


if __name__ == '__main__':
    df_movies, cosine_sim = load_and_preprocess_static_data()
    initialize_user_data()
    
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        # Truyền df_movies và cosine_sim vào authentication_page
        authentication_page(df_movies, cosine_sim)
