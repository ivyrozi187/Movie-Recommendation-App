import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ==============================================================================
# 1. CẤU HÌNH TRANG & CSS PHONG CÁCH NETFLIX
# ==============================================================================
st.set_page_config(
    page_title="DreamStream | AI Movie RecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để tạo giao diện giống Netflix
st.markdown("""
<style>
    /* Nền tối toàn trang */
    .main {
        background-color: #141414;
        color: white;
    }
    
    /* Sidebar đen sâu */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }

    /* Tiêu đề đỏ đặc trưng Netflix */
    h1, h2, h3 {
        color: #E50914 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Hiệu ứng Movie Card khi di chuột vào */
    .movie-container {
        position: relative;
        transition: transform .3s ease;
        cursor: pointer;
        margin-bottom: 10px;
    }
    
    .movie-container:hover {
        transform: scale(1.05);
        z-index: 10;
    }

    .poster-img {
        border-radius: 4px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.8);
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
    }

    /* Nút bấm đỏ Netflix */
    div.stButton > button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        height: 3em !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: #ff1f1f !important;
        box-shadow: 0 0 10px #ff4b4b;
    }

    /* Input và Selectbox style tối */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* Tab navigation */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    # Giả định file có sẵn, bạn hãy kiểm tra lại đường dẫn file của bạn
    try:
        movies = pd.read_csv("data_phim_full_images.csv")
        users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")
    except:
        st.error("Không tìm thấy file CSV. Vui lòng kiểm tra lại file data.")
        st.stop()

    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')
    
    movies['combined_features'] = (
        movies['Tên phim'] + " " + 
        movies['Đạo diễn'] + " " + 
        movies['Thể loại phim']
    )

    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    users['history_list'] = users['5 phim coi gần nhất'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    all_genres = set()
    for genres in movies['Thể loại phim']:
        for g in genres.split(','):
            all_genres.add(g.strip())
    
    return movies, users, cosine_sim, sorted(list(all_genres))

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. CÁC HÀM CHỨC NĂNG CỐT LÕI
# ==============================================================================

def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, exclude=None):
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    if exclude is None: exclude = []
    
    if not indices:
        popular_movies = movies_df.drop(exclude, errors='ignore').sort_values(by='Độ phổ biến', ascending=False)
        recs = popular_movies.head(top_k)
        return recs, recs.index.tolist()

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    scores_with_idx = list(enumerate(final_scores))
    scores_with_idx = sorted(scores_with_idx, key=lambda x: x[1], reverse=True)
    
    final_indices = []
    for i, score in scores_with_idx:
        if i not in indices and i not in exclude:
            final_indices.append(i)
            if len(final_indices) >= top_k:
                break
    
    return movies_df.iloc[final_indices], final_indices

def search_movie_func(query):
    return movies_df[movies_df['Tên phim'].str.contains(query, case=False, na=False)]

def get_genre_recommendations(selected_genres, top_k=10, exclude=None):
    if not selected_genres: return pd.DataFrame()
    if exclude is None: exclude = []

    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    if exclude:
        filtered = filtered.drop(exclude, errors='ignore')

    if filtered.empty: return pd.DataFrame()
    return filtered.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

def draw_user_charts(history_titles):
    if not history_titles:
        st.warning("Chưa có dữ liệu lịch sử.")
        return

    genres_count = []
    for title in history_titles:
        movie_row = movies_df[movies_df['Tên phim'] == title]
        if not movie_row.empty:
            g_str = movie_row.iloc[0]['Thể loại phim']
            g_list = [x.strip() for x in g_str.split(',')]
            genres_count.extend(g_list)
    
    if not genres_count: return

    counts = Counter(genres_count)
    df_chart = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df_chart.columns = ['Thể loại', 'Số phim đã xem']
    df_chart = df_chart.sort_values(by='Số phim đã xem', ascending=False)

    tab1, tab2 = st.tabs(["📊 Phân bố (%)", "📈 Số lượng"])
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='#141414')
        ax1.pie(df_chart['Số phim đã xem'], labels=df_chart['Thể loại'], autopct='%1.1f%%', textprops={'color':"w"})
        st.pyplot(fig1)
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#141414')
        sns.barplot(x='Số phim đã xem', y='Thể loại', data=df_chart, palette='Reds_r')
        ax2.tick_params(colors='white')
        st.pyplot(fig2)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI) - PHONG CÁCH NETFLIX
# ==============================================================================

if 'user_mode' not in st.session_state: st.session_state.user_mode = None
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'user_genres' not in st.session_state: st.session_state.user_genres = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 40px;'>DREAMSTREAM</h1>", unsafe_allow_html=True)
    
    if st.session_state.user_mode == 'member':
        st.success(f"Chào mừng trở lại, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Khám phá", ["Đề xuất AI", "Tìm kiếm Phim", "Thể loại Yêu thích", "Thống kê"])
        if st.button("Đăng xuất"):
            st.session_state.user_mode = None
           
    elif st.session_state.user_mode in ['guest', 'register']:
        menu = st.radio("Khám phá", ["Theo Thể loại"])
        if st.button("Thoát"):
            st.session_state.user_mode = None

    
    else:
        menu = "Login"

# --- Main Content ---
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])
    with tab1:
        username = st.text_input("Tên đăng nhập")
        if st.button("Truy cập"):
            user_row = users_df[users_df['Tên người dùng'] == username]
            if not user_row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = user_row.iloc[0]
             
            else: st.error("Sai thông tin!")
    # ... logic Register và Guest giữ nguyên như cũ của bạn ...
    with tab2:
        new_user = st.text_input("Tạo tên mới")
        selected_g = st.multiselect("Thể loại thích:", ALL_GENRES)
        if st.button("Tạo tài khoản"):
            st.session_state.user_mode = 'register'; st.session_state.user_genres = selected_g; 
    with tab3:
        guest_g = st.multiselect("Muốn xem gì hôm nay?", ALL_GENRES)
        if st.button("Xem ngay"):
            st.session_state.user_mode = 'guest'; st.session_state.user_genres = guest_g; 

elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']
    
    if menu == "Đề xuất AI":
        # HIỂN THỊ BANNER HERO GIỐNG NETFLIX
        recs, idxs = get_ai_recommendations(user_history, top_k=5)
        if not recs.empty:
            hero = recs.iloc[0]
            st.markdown(f"""
                <div style="background-image: linear-gradient(to right, #141414, rgba(20,20,20,0)), url('{hero['Link Poster']}'); 
                            background-size: cover; background-position: center 20%; height: 450px; border-radius: 10px; padding: 60px; display: flex; flex-direction: column; justify-content: center;">
                    <h1 style='font-size: 4em; margin: 0;'>{hero['Tên phim']}</h1>
                    <p style='max-width: 600px; font-size: 1.2em;'>{hero['Mô tả'][:200]}...</p>
                    <div style='display: flex; gap: 10px;'>
                        <button style='padding: 10px 30px; border-radius: 4px; border: none; font-weight: bold; cursor: pointer;'>▶ Phát</button>
                        <button style='padding: 10px 30px; border-radius: 4px; border: none; background: rgba(109, 109, 110, 0.7); color: white; font-weight: bold; cursor: pointer;'>ⓘ Thông tin</button>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### Danh sách gợi ý cho bạn")
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.markdown(f"""<div class="movie-container"><img src="{row['Link Poster']}" class="poster-img"></div>""", unsafe_allow_html=True)
                st.write(f"**{row['Tên phim']}**")
                with st.expander("Chi tiết"):
                    st.caption(f"🎬 {row['Đạo diễn']} | ⭐ {row['Độ phổ biến']}")
                    st.write(row['Mô tả'][:100] + "...")

    elif menu == "Tìm kiếm Phim":
        st.header("🔍 Tìm kiếm")
        query = st.text_input("Nhập tên phim...")
        if query:
            results = search_movie_func(query)
            if not results.empty:
                cols = st.columns(5)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 5]:
                        st.markdown(f"""<div class="movie-container"><img src="{row['Link Poster']}" class="poster-img"></div>""", unsafe_allow_html=True)
                        st.write(f"**{row['Tên phim']}**")
            else: st.warning("Không tìm thấy!")

    elif menu == "Thể loại Yêu thích":
        st.header("❤️ Dành cho fan của bạn")
        # Giữ nguyên logic get_genre_recommendations của bạn nhưng bọc trong HTML style
        fav_movie = st.session_state.current_user.get('Phim yêu thích nhất', '')
        if fav_movie:
            row = movies_df[movies_df['Tên phim'] == fav_movie]
            if not row.empty:
                fav_genres = [x.strip() for x in row.iloc[0]['Thể loại phim'].split(',')]
                recs = get_genre_recommendations(fav_genres, top_k=10)
                cols = st.columns(5)
                for i, (idx, r) in enumerate(recs.iterrows()):
                    with cols[i % 5]:
                        st.markdown(f"""<div class="movie-container"><img src="{r['Link Poster']}" class="poster-img"></div>""", unsafe_allow_html=True)
                        st.write(f"**{r['Tên phim']}**")

    elif menu == "Thống kê":
        draw_user_charts(user_history)

elif st.session_state.user_mode in ['guest', 'register']:
    selected_g = st.session_state.user_genres
    st.header(f"📂 Khám phá {selected_g[0] if selected_g else ''}")
    # Logic guest giữ nguyên, hiển thị card giống Member phía trên
    sub_genre = st.selectbox("Chọn thể loại cụ thể:", selected_g)
    recs = get_genre_recommendations([sub_genre], top_k=15)
    if not recs.empty:
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.markdown(f"""<div class="movie-container"><img src="{row['Link Poster']}" class="poster-img"></div>""", unsafe_allow_html=True)
                st.write(f"**{row['Tên phim']}**")
