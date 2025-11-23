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

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- CONSTANT ---
GUEST_USER = "Guest_ZeroClick" 

# Bản đồ ánh xạ chủ đề hiển thị (Dùng cho cả Guest mode và Đăng ký mới)
# Cấu trúc: Tên chủ đề -> {Genres con, Màu sắc, Gradient}
INTRO_TOPICS = {
    "Marvel": {"genres": ["Action", "Sci-Fi", "Fantasy"], "color": "#5c67e2", "gradient": "#7983e2"},
    "4K": {"genres": ["Action", "Adventure", "Sci-Fi"], "color": "#7e8399", "gradient": "#8d90a7"},
    "Sitcom": {"genres": ["Comedy", "TV Movie"], "color": "#35a371", "gradient": "#42b883"},
    "Lồng Tiếng Cực Mạnh": {"genres": ["Action", "Adventure", "Drama"], "color": "#9665d9", "gradient": "#a881e6"},
    "Xuyên Không": {"genres": ["Sci-Fi", "Fantasy", "Adventure"], "color": "#d18c69", "gradient": "#e0a17f"},
    "Cổ Trang": {"genres": ["History", "War", "Drama"], "color": "#a54545", "gradient": "#b85c5c"},
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
    
    st.markdown("### Bạn đang quan tâm gì?")
    st.caption("Chọn các chủ đề bạn thích để chúng tôi xây dựng hồ sơ ban đầu:")

    # CSS cho thẻ Topic (giống hình ảnh)
    st.markdown("""
    <style>
        div[data-testid*="stButton"] > button {
             border: none; 
        }
    </style>
    """, unsafe_allow_html=True)

    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(3) # Chia lưới 3 cột
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        is_selected = topic in st.session_state['selected_reg_topics']
        
        # Style động: Nếu chọn thì có viền sáng/shadow, nếu không thì bình thường
        border_style = "border: 3px solid #f63366; box-shadow: 0 0 15px rgba(246, 51, 102, 0.6);" if is_selected else "border: none;"
        opacity = "1.0" if is_selected else "0.85"
        scale = "transform: scale(1.02);" if is_selected else ""
        
        # Tạo style riêng cho từng nút
        btn_style = f"""
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 12px;
            height: 100px;
            font-weight: bold;
            font-size: 1.1rem;
            width: 100%;
            margin-bottom: 10px;
            {border_style}
            opacity: {opacity};
            {scale}
            transition: all 0.2s ease-in-out;
        """

        with cols[i % 3]:
            # Nút bấm toggle
            st.button(
                f"{topic}\nXem chủ đề >", 
                key=f"reg_topic_{topic}", 
                on_click=toggle_reg_topic, 
                args=(topic,),
                use_container_width=True
            )
            
            # Inject CSS vào nút vừa tạo
            st.markdown(f"""
                <style>
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"] {{
                        {btn_style}
                    }}
                    /* Override hover effect */
                    div[data-testid="stButton"] button[key="reg_topic_{topic}"]:hover {{
                        opacity: 1.0;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                        {border_style}
                        color: white;
                    }}
                </style>
            """, unsafe_allow_html=True)


def register_new_user_form(df_movies):
    """Form đăng ký người dùng mới (Đã sửa đổi theo yêu cầu)."""
    st.header("📝 Đăng Ký Tài Khoản Mới")
    st.info("📢 Người dùng mới sẽ chỉ tồn tại trong phiên làm việc hiện tại.")

    df_users = st.session_state['df_users']
    
    # 1. Nhập tên người dùng
    username = st.text_input("Tên người dùng mới (Duy nhất):", key="reg_username").strip()

    st.write("---")

    # 2. Chọn chủ đề (Thay thế phần chọn thể loại và phim yêu thích cũ)
    # Lưu ý: Không dùng st.form bao quanh phần này để nút bấm tương tác được ngay
    draw_registration_topic_cards()
    
    selected_topics = list(st.session_state['selected_reg_topics'])
    
    st.write("")
    if selected_topics:
        st.success(f"✅ Đã chọn: {', '.join(selected_topics)}")
    else:
        st.warning("Vui lòng chọn ít nhất 1 chủ đề.")

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
            st.error("❌ Vui lòng chọn ít nhất 1 chủ đề quan tâm.")
            return
        
        # --- LOGIC MỚI: CHUYỂN ĐỔI TOPIC -> GENRES ---
        # Lấy tất cả genres từ các topic đã chọn để lưu vào hồ sơ
        mapped_genres = set()
        for topic in selected_topics:
            if topic in INTRO_TOPICS:
                mapped_genres.update(INTRO_TOPICS[topic]['genres'])
        
        final_genres_list = list(mapped_genres)
        
        # Tạo ID mới
        max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
        new_id = int(max_id) + 1
        
        # Lưu trữ
        new_user_data = {
            'ID': [new_id],
            'Tên người dùng': [username],
            # Lưu danh sách genres đã convert từ topics vào cột này
            '5 phim coi gần nhất': [str(final_genres_list)], 
            # Bỏ chọn phim yêu thích, lưu giá trị mặc định hoặc rỗng
            'Phim yêu thích nhất': [""] 
        }
        new_user_df = pd.DataFrame(new_user_data)
        
        st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
        
        st.session_state['logged_in_user'] = username
        st.balloons()
        st.success(f"🎉 Đăng ký thành công! Đã thiết lập hồ sơ theo sở thích: {', '.join(selected_topics)}.")
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
        register_new_user_form(df_movies)

# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ (GIỮ NGUYÊN)
# ==============================================================================

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

def plot_genre_popularity(movie_name, recommended_movies_df, df_movies, is_user_based=False):
    df_users = st.session_state['df_users']
    combined_df = recommended_movies_df.copy() 
    
    if is_user_based:
        user_row = df_users[df_users['Tên người dùng'] == st.session_state['logged_in_user']]
        user_genres_str = user_row['5 phim coi gần nhất'].iloc[0]
        user_genres_list = []
        try:
            user_genres_list = ast.literal_eval(user_genres_str)
            if not isinstance(user_genres_list, list): user_genres_list = []
        except:
            user_genres_list = [m.strip().strip("'") for m in user_genres_str.strip('[]').split(',') if m.strip()]
        
        genre_data_for_plot = []
        for genre in user_genres_list:
            avg_pop = df_movies[df_movies['Thể loại phim'].str.contains(genre, case=False, na=False)]['Độ phổ biến'].mean()
            genre_data_for_plot.append({'Tên phim': f'Hồ sơ: {genre}', 'Thể loại phim': genre, 'Độ phổ biến': avg_pop if pd.notna(avg_pop) else 0})

        watched_genres_df = pd.DataFrame(genre_data_for_plot)
        combined_df = pd.concat([watched_genres_df[['Thể loại phim', 'Độ phổ biến']], recommended_movies_df[['Thể loại phim', 'Độ phổ biến']]], ignore_index=True)
        title = f"Độ Phổ Biến Thể Loại (Hồ sơ {st.session_state['logged_in_user']} & Đề xuất)"

    else:
        if st.session_state['logged_in_user'] == GUEST_USER:
             title = "Độ Phổ Biến Thể Loại (Đề xuất Zero-Click)"
        else:
            movie_row = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()]
            if movie_row.empty: return
            combined_df = pd.concat([movie_row, recommended_movies_df], ignore_index=True)
            title = f"Độ Phổ Biến TB của Các Thể Loại Phim Liên Quan đến '{movie_name}'"

    genres_data = []
    combined_df = combined_df[['Thể loại phim', 'Độ phổ biến']].dropna()
    for index, row in combined_df.iterrows():
        genres_list = [g.strip() for g in row['Thể loại phim'].split(',') if g.strip()]
        for genre in genres_list:
            genres_data.append({'Thể loại': genre, 'Độ phổ biến': row['Độ phổ biến']})

    df_plot = pd.DataFrame(genres_data)
    if df_plot.empty: return
        
    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)
    overall_avg_pop = df_plot['Độ phổ biến'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'], color='skyblue', edgecolor='black', alpha=0.8)
    ax.axhline(overall_avg_pop, color='red', linestyle='--', linewidth=1.5, label=f'TB Tổng thể ({overall_avg_pop:.1f})')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, round(bar.get_height(), 1), ha='center', fontsize=10, weight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Thể loại phim")
    ax.set_ylabel("Độ Phổ Biến Trung Bình")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig) 

# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def draw_interest_cards_guest():
    """Giao diện thẻ cho chế độ Khách (Guest) - Chỉ chọn 1."""
    st.header("Bạn đang quan tâm gì?")
    st.markdown("Chọn một chủ đề để nhận đề xuất ngay:")
    
    st.markdown("""
    <style>
        div[data-testid="stButton"] button {
            border: none;
            transition: transform 0.2s;
        }
        div[data-testid="stButton"] button:hover {
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

    topics = list(INTRO_TOPICS.keys())
    cols = st.columns(3)
    
    for i, topic in enumerate(topics):
        data = INTRO_TOPICS[topic]
        btn_style = f"""
            background: linear-gradient(135deg, {data['color']}, {data['gradient']});
            color: white;
            border-radius: 12px;
            height: 120px;
            font-weight: bold;
            font-size: 1.2rem;
            width: 100%;
            margin-bottom: 15px;
        """
        with cols[i % 3]:
            st.button(f"{topic}\nXem chủ đề >", key=f"guest_{topic}", on_click=select_topic, args=(topic,), use_container_width=True)
            st.markdown(f"""<style>div[data-testid="stButton"] button[key="guest_{topic}"] {{ {btn_style} }}</style>""", unsafe_allow_html=True)

def main_page(df_movies, cosine_sim):
    is_guest = st.session_state['logged_in_user'] == GUEST_USER
    username_display = "Khách" if is_guest else st.session_state['logged_in_user']
    
    st.title(f"🎬 Chào mừng, {username_display}!")
    st.sidebar.title("Menu Đề Xuất")
    
    if is_guest:
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
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(st.session_state['last_guest_result']['Tên phim'].tolist())]
                    plot_genre_popularity(None, recommended_movies_info, df_movies, is_user_based=False)
            
            if st.sidebar.button("Đăng Xuất Khách", on_click=logout): pass

    else:
        df_users = st.session_state['df_users']
        menu_choice = st.sidebar.radio("Chọn chức năng:", ('Đề xuất theo Tên Phim', 'Đề xuất theo Hồ Sơ', 'Đăng Xuất'))

        if st.sidebar.button("Đăng Xuất", on_click=logout): pass 
        st.sidebar.write("-" * 20)

        if menu_choice == 'Đề xuất theo Tên Phim':
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
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(st.session_state['last_sim_result']['Tên phim'].tolist())]
                    plot_genre_popularity(st.session_state['last_sim_movie'], recommended_movies_info, df_movies, is_user_based=False)

        elif menu_choice == 'Đề xuất theo Hồ Sơ':
            st.header("2️⃣ Đề xuất theo Hồ sơ")
            username = st.session_state['logged_in_user']
            user_row = df_users[df_users['Tên người dùng'] == username]
            
            if st.button("Tìm Đề Xuất Hồ Sơ", key="find_profile"):
                recommendations = get_recommendations(username, df_movies)
                if not recommendations.empty:
                    st.session_state['last_profile_recommendations'] = recommendations
                    st.session_state['show_profile_plot'] = True 
                else:
                    st.warning("Chưa đủ dữ liệu để đề xuất.")
                st.rerun()

            if not st.session_state['last_profile_recommendations'].empty:
                st.subheader(f"✅ Đề xuất Dành Riêng Cho Bạn:")
                st.dataframe(st.session_state['last_profile_recommendations'], use_container_width=True)
                if st.checkbox("📊 Hiển thị Biểu đồ", value=st.session_state['show_profile_plot'], key="plot_profile_check"):
                    recommended_movies_info = df_movies[df_movies['Tên phim'].isin(st.session_state['last_profile_recommendations']['Tên phim'].tolist())]
                    plot_genre_popularity(None, recommended_movies_info, df_movies, is_user_based=True)

if __name__ == '__main__':
    df_movies, cosine_sim = load_and_preprocess_static_data()
    initialize_user_data()
    
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies)
