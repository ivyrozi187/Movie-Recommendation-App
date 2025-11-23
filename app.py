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

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" # Định danh cho người dùng chế độ Khách

# Bản đồ ánh xạ chủ đề hiển thị (như trong ảnh) sang thể loại (genres) và màu sắc (CSS)
INTRO_TOPICS = {
    "Marvel": {"genres": ["Action", "Sci-Fi", "Fantasy"], "color": "#7983e2", "gradient": "#5c67e2"},
    "4K": {"genres": ["Action", "Adventure", "Sci-Fi"], "color": "#8d90a7", "gradient": "#7e8399"},
    "Sitcom": {"genres": ["Comedy", "TV Movie"], "color": "#42b883", "gradient": "#35a371"},
    "Lồng Tiếng Cực Mạnh": {"genres": ["Action", "Adventure", "Drama"], "color": "#a881e6", "gradient": "#9665d9"},
    "Xuyên Không": {"genres": ["Sci-Fi", "Fantasy", "Adventure"], "color": "#e0a17f", "gradient": "#d18c69"},
    "Cổ Trang": {"genres": ["History", "War", "Drama"], "color": "#b85c5c", "gradient": "#a54545"},
}

# --- KHỞI TẠO BIẾN TRẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'

# Khởi tạo các biến để lưu kết quả và trạng thái hiển thị biểu đồ
if 'last_sim_result' not in st.session_state: st.session_state['last_sim_result'] = pd.DataFrame()
if 'last_sim_movie' not in st.session_state: st.session_state['last_sim_movie'] = None
if 'show_sim_plot' not in st.session_state: st.session_state['show_sim_plot'] = False

if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False

# Biến trạng thái mới cho chức năng Zero-Click với Card
if 'selected_intro_topics' not in st.session_state: st.session_state['selected_intro_topics'] = []
if 'last_guest_result' not in st.session_state: st.session_state['last_guest_result'] = pd.DataFrame()
if 'show_guest_plot' not in st.session_state: st.session_state['show_guest_plot'] = False


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
    
@st.cache_resource 
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
        df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
        mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
        df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)
        
        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        # 3. Tiền xử lý cho Zero-Click (Recency and Global Genre Popularity)
        
        # 3a. Tính điểm mới nhất (Recency) - Giả định có cột 'Năm phát hành'
        if 'Năm phát hành' in df_movies.columns:
            df_movies['Năm phát hành'] = pd.to_numeric(df_movies['Năm phát hành'], errors='coerce').fillna(pd.Timestamp('now').year)
            # Chuẩn hóa điểm Recency (Phim mới nhất có điểm cao nhất)
            max_year = df_movies['Năm phát hành'].max()
            min_year = df_movies['Năm phát hành'].min()
            if max_year > min_year:
                 df_movies['recency_score'] = (df_movies['Năm phát hành'] - min_year) / (max_year - min_year)
            else:
                 df_movies['recency_score'] = 0.5 # Default score if all years are the same
        else:
            # Nếu cột 'Năm phát hành' không tồn tại, dùng điểm phổ biến làm placeholder
            df_movies['recency_score'] = df_movies["popularity_norm"] * 0.1 

        # 3b. Tính điểm phổ biến thể loại toàn cầu (Global Genre Popularity)
        genres_pop = {}
        for index, row in df_movies.iterrows():
            popularity = row['Độ phổ biến']
            for genre in row['Thể loại phim'].split(','):
                genre = genre.strip()
                if genre:
                    genres_pop.setdefault(genre, []).append(popularity)
        
        global_genre_popularity = {g: sum(p)/len(p) for g, p in genres_pop.items() if len(p) > 0}
        
        # Chuẩn hóa điểm phổ biến thể loại
        max_pop = max(global_genre_popularity.values()) if global_genre_popularity else 1
        normalized_genre_pop = {g: p / max_pop for g, p in global_genre_popularity.items()}

        df_movies['global_genre_score'] = df_movies['Thể loại phim'].apply(
            lambda x: max([normalized_genre_pop.get(g.strip(), 0) for g in x.split(',')], default=0) if x else 0
        )

        return df_movies, cosine_sim_matrix 

    except Exception as e:
        st.error(f"LỖI TẢI HOẶC XỬ LÝ DỮ LIỆU TĨNH: {e}. Vui lòng kiểm tra các file CSV.")
        return pd.DataFrame(), np.array([[]])


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu người dùng vào Session State."""
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            df_users.columns = [col.strip() for col in df_users.columns]
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
            
            df_users = df_users.dropna(subset=['ID'])
        except Exception:
            df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất'])

        st.session_state['df_users'] = df_users
    
    return st.session_state['df_users']

def get_unique_movie_titles(df_movies):
    """Lấy danh sách các tên phim duy nhất."""
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

# --- CALLBACK FUNCTIONS ---
def set_auth_mode(mode):
    """Hàm callback để chuyển đổi giữa Đăng nhập và Đăng ký."""
    st.session_state['auth_mode'] = mode
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()

def login_as_guest():
    """Hàm callback để đăng nhập dưới dạng Khách (Zero-Click)."""
    st.session_state['logged_in_user'] = GUEST_USER
    st.session_state['auth_mode'] = 'login' 
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.session_state['selected_intro_topics'] = [] # Reset topic selection
    st.session_state['last_guest_result'] = pd.DataFrame() # Reset results
    st.rerun() # Chạy lại để chuyển sang main_page

def logout():
    """Hàm callback để Đăng xuất."""
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    st.session_state['selected_intro_topics'] = [] # Reset topic selection
    st.session_state['last_guest_result'] = pd.DataFrame() # Reset results

# Hàm callback khi bấm vào thẻ chủ đề
def select_topic(topic_key):
    """Lưu chủ đề đã chọn và kích hoạt tìm kiếm."""
    st.session_state['selected_intro_topics'] = [topic_key]
    st.session_state['last_guest_result'] = pd.DataFrame() # Xóa kết quả cũ
    st.rerun()
# ---------------------------

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
            if not username:
                st.error("Vui lòng nhập tên người dùng.")
                return
            
            if username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên người dùng '{username}' đã tồn tại.")
                return
            
            if len(recent_list_raw) < 5:
                st.warning("Vui lòng chọn tối thiểu 5 phim đã xem gần nhất.")
                return
            
            max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
            new_id = int(max_id) + 1
            
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': [str(recent_list_raw)], 
                'Phim yêu thích nhất': [favorite_movie]
            }
            new_user_df = pd.DataFrame(new_user_data)
            
            st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
            
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
            if username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Đăng nhập thành công! Chào mừng, {username}.")
                st.rerun() 
            else:
                st.error("❌ Tên người dùng không tồn tại.")

def authentication_page(df_movies):
    """Trang Xác thực (chọn Đăng nhập hoặc Đăng ký) và Zero-Click."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',))
    with col2:
        st.button("Đăng Ký", key="btn_register", on_click=set_auth_mode, args=('register',))

    st.write("---")
    st.subheader("Hoặc:")
    # Thay đổi nút Guest để sử dụng callback
    st.button("🚀 Thử Dùng Với Chế Độ Khách (Zero-Click)", key="btn_guest", on_click=login_as_guest)
    st.caption("Bạn sẽ được chuyển đến trang chọn sở thích để nhận đề xuất chung ban đầu.")

    if st.session_state['auth_mode'] == 'login':
        login_form()
    
    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies)

# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ
# ==============================================================================

def get_zero_click_recommendations(df_movies, selected_genres, num_recommendations=15):
    """
    Đề xuất 'Zero-Click' có cá nhân hóa dựa trên thể loại đã chọn (selected_genres)
    """
    
    # Đặt trọng số cơ bản
    WEIGHT_POPULARITY = 0.50 
    WEIGHT_RECENCY = 0.25
    WEIGHT_GENRE_POPULARITY = 0.25
    WEIGHT_TOPIC_BOOST = 0.50 # Trọng số điểm boost dựa trên lựa chọn chủ đề
    
    if df_movies.empty or 'popularity_norm' not in df_movies.columns:
        return pd.DataFrame()
    
    df = df_movies.copy()
    
    # 1. Tính điểm Zero-Click cơ bản
    df['base_zero_click_score'] = (
        WEIGHT_POPULARITY * df['popularity_norm'] +
        WEIGHT_RECENCY * df['recency_score'] +
        WEIGHT_GENRE_POPULARITY * df['global_genre_score']
    )
    
    # 2. Áp dụng điểm BOOST từ lựa chọn thẻ
    if selected_genres:
        # Tạo hàm tính điểm boost (điểm 1 nếu phim có chứa bất kỳ genre nào đã chọn)
        def calculate_boost(parsed_genres):
            return 1 if not parsed_genres.isdisjoint(set(selected_genres)) else 0
        
        df['topic_boost'] = df['parsed_genres'].apply(calculate_boost)
        
        # Điểm tổng cuối cùng: Base Score + (Boost Score * Trọng số Boost)
        df['combined_zero_click_score'] = df['base_zero_click_score'] + (df['topic_boost'] * WEIGHT_TOPIC_BOOST)
    else:
        # Nếu không chọn gì, chỉ dùng Base Score
        df['combined_zero_click_score'] = df['base_zero_click_score']

    recommended_df = df.sort_values(
        by='combined_zero_click_score',
        ascending=False
    )
    
    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'combined_zero_click_score']].head(num_recommendations)


def get_recommendations(username, df_movies, num_recommendations=10):
    """Đề xuất phim dựa trên 5 phim người dùng xem gần nhất và sở thích thể loại."""
    
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
    watched_list = []
    
    try:
        watched_list = ast.literal_eval(watched_movies_str)
        if not isinstance(watched_list, list): watched_list = []
    except (ValueError, SyntaxError):
        watched_list = [m.strip().strip("'") for m in watched_movies_str.strip('[]').split(',') if m.strip()]
    
    watched_list = [str(item) for item in watched_list if str(item).strip()]


    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    watched_and_favorite = set(watched_list + [favorite_movie])

    watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
    user_genres = set()
    for genres in watched_genres['parsed_genres']:
        user_genres.update(genres)

    if not user_genres: 
        return pd.DataFrame()

    candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()

    def calculate_score(candidate_genres):
        return len(candidate_genres.intersection(user_genres))

    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(calculate_score)

    recommended_df = candidate_movies.sort_values(
        by=['Similarity_Score', 'Độ phổ biến'],
        ascending=[False, False]
    )

    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)

def get_movie_index(movie_name, df_movies):
    """Tìm chỉ mục của phim trong DataFrame."""
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
        return idx
    except IndexError:
        return -1

def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    """Đề xuất phim dựa trên sự kết hợp giữa độ giống (sim) và độ phổ biến (pop)."""
    
    if cosine_sim.size == 0 or df_movies.empty:
        st.warning("Dữ liệu phim chưa được tải hoặc bị lỗi. Không thể thực hiện đề xuất.")
        return pd.DataFrame()
        
    idx = get_movie_index(movie_name, df_movies)
    if idx == -1: return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])

    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    df_result['weighted_score'] = (
            weight_sim * df_result['similarity'] +
            weight_pop * df_result['popularity_norm']
    )

    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)

    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)

def plot_genre_popularity(movie_name, recommended_movies_df, df_movies, is_user_based=False):
    """Vẽ biểu đồ so sánh ĐỘ PHỔ BIẾN TRUNG BÌNH của các thể loại liên quan."""

    df_users = st.session_state['df_users']
    combined_df = recommended_movies_df.copy() 
    
    if is_user_based:
        user_row = df_users[df_users['Tên người dùng'] == st.session_state['logged_in_user']]
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = []
        try:
            watched_list = ast.literal_eval(watched_movies_str)
            if not isinstance(watched_list, list): watched_list = []
        except:
            watched_list = [m.strip().strip("'") for m in watched_movies_str.strip('[]').split(',') if m.strip()]
        
        watched_list = [str(item) for item in watched_list if str(item).strip()]
            
        watched_df = df_movies[df_movies['Tên phim'].isin(watched_list)]
        
        combined_df = pd.concat([watched_df, recommended_movies_df], ignore_index=True)
        title = f"Độ Phổ Biến Thể Loại (Hồ sơ {st.session_state['logged_in_user']} & Đề xuất)"

    else:
        # Nếu là Zero-Click, không có phim gốc để so sánh, chỉ lấy recommended_movies_df
        if st.session_state['logged_in_user'] == GUEST_USER:
             title = "Độ Phổ Biến Thể Loại (Đề xuất Zero-Click)"
        else:
            # Dành cho đề xuất Content-based thông thường
            movie_row = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()]
            if movie_row.empty: 
                st.error(f"Không tìm thấy thông tin phim gốc '{movie_name}' để so sánh.")
                return
            combined_df = pd.concat([movie_row, recommended_movies_df], ignore_index=True)
            title = f"Độ Phổ Biến TB của Các Thể Loại Phim Liên Quan đến '{movie_name}'"

    genres_data = []
    combined_df = combined_df[['Thể loại phim', 'Độ phổ biến']].dropna()
    
    for index, row in combined_df.iterrows():
        genres_list = [g.strip() for g in row['Thể loại phim'].split(',') if g.strip()]
        for genre in genres_list:
            genres_data.append({
                'Thể loại': genre,
                'Độ phổ biến': row['Độ phổ biến']
            })

    df_plot = pd.DataFrame(genres_data)
    
    if df_plot.empty:
        st.warning("Không đủ dữ liệu thể loại (Thường do thông tin phim bị thiếu thể loại) để vẽ biểu đồ. Vui lòng kiểm tra file `movie_info_1000.csv`.")
        return
        
    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)
    overall_avg_pop = df_plot['Độ phổ biến'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'], 
                  color='skyblue', edgecolor='black', alpha=0.8)

    ax.axhline(overall_avg_pop, color='red', linestyle='--', linewidth=1.5, 
               label=f'TB Tổng thể ({overall_avg_pop:.1f})')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 1), ha='center', fontsize=10, weight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Thể loại phim")
    ax.set_ylabel("Độ Phổ Biến Trung Bình (Popularity Score)")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig) 

# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def draw_interest_cards():
    """Vẽ giao diện chọn thẻ chủ đề."""
    st.header("Bạn đang quan tâm gì?")
    st.markdown("Chọn một hoặc nhiều chủ đề để nhận đề xuất ban đầu được cá nhân hóa:", unsafe_allow_html=True)
    
    # CSS để tạo kiểu thẻ
    st.markdown("""
    <style>
        .interest-card {
            border-radius: 15px;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .interest-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        .interest-card h3 {
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 10px;
        }
        .interest-card .details {
            font-size: 0.9rem;
            opacity: 0.8;
        }
    </style>
    """, unsafe_allow_html=True)

    topics = list(INTRO_TOPICS.keys())
    
    # Tạo layout 3 cột, lặp lại cho các chủ đề
    cols = st.columns(3)
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        
        # HTML cho mỗi thẻ (sử dụng background gradient và nút ẩn)
        card_html = f"""
        <div class="interest-card" style="background: linear-gradient(135deg, {data['color']}, {data['gradient']});">
            <h3>{topic}</h3>
            <div class="details">Xem chủ đề ></div>
        </div>
        """
        
        # Sử dụng st.button để tạo sự kiện click
        # Đặt button trên st.markdown để nó thực sự kích hoạt Streamlit Rerun
        with cols[i % 3]:
            # Hiển thị thẻ bằng HTML
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Tạo nút ẩn (zero-height/opacity) để bắt sự kiện click
            if st.button(f"Chọn {topic}", key=f"select_{topic}", use_container_width=True):
                select_topic(topic)


def main_page(df_movies, cosine_sim):
    
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username_display = "Khách" if is_guest else st.session_state['logged_in_user']
    
    st.title(f"🎬 Chào mừng, {username_display}!")
    
    st.sidebar.title("Menu Đề Xuất")
    
    if is_guest:
        # --- CHẾ ĐỘ KHÁCH (ZERO-CLICK) ---
        st.header("🔥 Đề xuất Zero-Click (Dựa trên Xu hướng Toàn cầu)")

        # 1. BƯỚC LỰA CHỌN CHỦ ĐỀ (Hiển thị nếu chưa chọn)
        if not st.session_state['selected_intro_topics']:
            draw_interest_cards()
            
            # Luôn có nút Đăng xuất cho Guest
            if st.sidebar.button("Đăng Xuất Khách", key="logout_guest_btn", on_click=logout):
                pass
            
            return # Dừng ở đây để chờ người dùng chọn
        
        # 2. BƯỚC HIỂN THỊ KẾT QUẢ (Nếu đã chọn chủ đề)
        else:
            selected_topics = st.session_state['selected_intro_topics']
            selected_genre_list = []
            for topic in selected_topics:
                selected_genre_list.extend(INTRO_TOPICS.get(topic, {}).get("genres", []))
            
            topic_names = ", ".join(selected_topics)
            st.info(f"Đề xuất đang được cá nhân hóa dựa trên chủ đề bạn đã chọn: **{topic_names}**.")
            
            # Tự động tìm kiếm nếu chưa có kết quả
            if st.session_state['last_guest_result'].empty:
                zero_click_results = get_zero_click_recommendations(df_movies, selected_genre_list, num_recommendations=15)
                
                if not zero_click_results.empty:
                    st.session_state['last_guest_result'] = zero_click_results
                    st.session_state['show_guest_plot'] = True
                else:
                    st.session_state['last_guest_result'] = pd.DataFrame()
                    st.session_state['show_guest_plot'] = False
                    st.warning("⚠️ Không thể tạo đề xuất Zero-Click. Vui lòng kiểm tra dữ liệu.")
            
            # Hiển thị kết quả và biểu đồ
            if not st.session_state['last_guest_result'].empty:
                zero_click_results = st.session_state['last_guest_result']
                st.subheader("✅ 15 Đề xuất Giới thiệu Tốt nhất Dành Cho Bạn:")
                st.dataframe(zero_click_results, use_container_width=True)
                
                show_plot_guest = st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", 
                                                value=st.session_state['show_guest_plot'],
                                                key="plot_guest_check")
                
                if show_plot_guest:
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(zero_click_results['Tên phim'].tolist())]
                    plot_genre_popularity(None, recommended_movies_info, df_movies, is_user_based=False)
            
            # Nút Đăng xuất ở sidebar
            if st.sidebar.button("Đăng Xuất Khách", key="logout_guest_btn", on_click=logout):
                pass

    else:
        # --- CHẾ ĐỘ NGƯỜI DÙNG ĐĂNG NHẬP ---
        df_users = st.session_state['df_users']
        
        menu_choice = st.sidebar.radio(
            "Chọn chức năng:",
            ('Đề xuất theo Tên Phim', 'Đề xuất theo Hồ Sơ', 'Đăng Xuất')
        )

        if st.sidebar.button("Đăng Xuất", key="logout_btn", on_click=logout):
            pass 
            
        st.sidebar.write("-" * 20)

        if menu_choice == 'Đề xuất theo Tên Phim':
            st.header("1️⃣ Đề xuất dựa trên Nội dung (TF-IDF)")
            
            movie_titles_list = get_unique_movie_titles(df_movies)
            
            default_movie_name = st.session_state['last_sim_movie'] if st.session_state['last_sim_movie'] in movie_titles_list else movie_titles_list[0]
            movie_name = st.selectbox("🎥 Chọn tên phim bạn yêu thích:", options=movie_titles_list, index=movie_titles_list.index(default_movie_name))
            
            weight_sim = st.slider("⚖️ Trọng số Độ giống (Similarity)", 0.0, 1.0, 0.7, 0.1)
            weight_pop = 1 - weight_sim
            
            if st.button("Tìm Đề Xuất", key="find_sim"):
                result = recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim)
                
                if not result.empty:
                    st.session_state['last_sim_result'] = result
                    st.session_state['last_sim_movie'] = movie_name
                    st.session_state['show_sim_plot'] = True 
                else:
                    st.session_state['last_sim_result'] = pd.DataFrame()
                    st.session_state['show_sim_plot'] = False
                    st.warning("⚠️ Không tìm thấy đề xuất hoặc phim gốc không tồn tại.")
                st.rerun() 

            if not st.session_state['last_sim_result'].empty:
                result = st.session_state['last_sim_result']
                movie_name_for_display = st.session_state['last_sim_movie']

                st.subheader(f"🎬 10 Đề xuất phim dựa trên '{movie_name_for_display}':")
                st.dataframe(result, use_container_width=True)

                show_plot = st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", 
                                        value=st.session_state['show_sim_plot'], 
                                        key="plot_sim_check")

                if show_plot:
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(result['Tên phim'].tolist())]
                    plot_genre_popularity(movie_name_for_display, recommended_movies_info, df_movies, is_user_based=False)

        elif menu_choice == 'Đề xuất theo Hồ Sơ':
            st.header("2️⃣ Đề xuất dựa trên Hồ sơ Người dùng")
            
            username = st.session_state['logged_in_user']
            user_row = df_users[df_users['Tên người dùng'] == username]
            
            if user_row.empty:
                st.error("LỖI: Không tìm thấy hồ sơ người dùng trong phiên. Vui lòng đăng ký lại.")
                return

            recent_films_str = user_row['5 phim coi gần nhất'].iloc[0]
            recent_films = []
            try:
                recent_films = ast.literal_eval(recent_films_str)
                if not isinstance(recent_films, list): recent_films = []
            except:
                recent_films = [m.strip().strip("'") for m in recent_films_str.strip('[]').split(',') if m.strip()]
            
            recent_films_display = ', '.join([str(item) for item in recent_films if str(item).strip()])

            st.info(f"5 Phim đã xem gần nhất: {recent_films_display}")
            
            if st.button("Tìm Đề Xuất Hồ Sơ", key="find_profile"):
                recommendations = get_recommendations(username, df_movies, num_recommendations=10)

                if not recommendations.empty:
                    st.session_state['last_profile_recommendations'] = recommendations
                    st.session_state['show_profile_plot'] = True 
                else:
                    st.session_state['last_profile_recommendations'] = pd.DataFrame()
                    st.session_state['show_profile_plot'] = False
                    st.warning("⚠️ Không có đề xuất nào được tạo. Kiểm tra dữ liệu thể loại phim đã xem.")
                st.rerun() 

            if not st.session_state['last_profile_recommendations'].empty:
                recommendations = st.session_state['last_profile_recommendations']

                st.subheader(f"✅ 10 Đề xuất Phim Dành Cho Bạn:")
                st.dataframe(recommendations, use_container_width=True)
                
                show_plot_profile = st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", 
                                                value=st.session_state['show_profile_plot'],
                                                key="plot_profile_check")
                
                if show_plot_profile:
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(recommendations['Tên phim'].tolist())]
                    plot_genre_popularity(None, 
                                          recommended_movies_info, 
                                          df_movies, is_user_based=True)


# ==============================================================================
# V. CHẠY ỨNG DỤNG CHÍNH
# ==============================================================================

if __name__ == '__main__':
    # 1. Tải dữ liệu tĩnh (Chỉ chạy 1 lần)
    df_movies, cosine_sim = load_and_preprocess_static_data()
    
    # 2. Khởi tạo dữ liệu người dùng (Sẽ được cập nhật khi đăng ký)
    initialize_user_data()
    
    # 3. Định tuyến trang
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies)
