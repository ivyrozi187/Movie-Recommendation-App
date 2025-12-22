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
# 1. CẤU HÌNH TRANG & CSS
# ==============================================================================
st.set_page_config(
    page_title="Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .refresh-button>button {
        background-color: #2e7d32 !important; /* Màu xanh lá cho nút làm mới */
        margin-bottom: 20px;
    }
    .movie-card {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")

    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')
    
    movies['combined_features'] = (movies['Tên phim'] + " " + movies['Đạo diễn'] + " " + movies['Thể loại phim'])

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
# 3. CÁC HÀM CHỨC NĂNG CỐT LÕI (CẬP NHẬT RANDOM)
# ==============================================================================

def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, seed=42):
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])
    
    if not indices:
        return movies_df.sample(n=top_k, random_state=seed)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)
    
    scores_with_idx = list(enumerate(final_scores))
    # Lọc bỏ phim đã xem và lấy top 50 phim tiềm năng nhất để random trong đó
    potential_indices = [i[0] for i in sorted(scores_with_idx, key=lambda x: x[1], reverse=True) if i[0] not in indices]
    
    # Lấy top 30 phim tốt nhất rồi chọn ngẫu nhiên 10 phim từ đó để tạo sự mới mẻ
    top_potential = potential_indices[:30]
    import random
    random.seed(seed)
    selected_indices = random.sample(top_potential, min(top_k, len(top_potential)))
    
    return movies_df.iloc[selected_indices]

def get_genre_recommendations(selected_genres, top_k=10, seed=42):
    if not selected_genres:
        return pd.DataFrame()
    
    pattern = '|'.join(selected_genres)
    filtered = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Random từ danh sách các phim thuộc thể loại đó
    return filtered.sample(n=min(top_k, len(filtered)), random_state=seed)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []
if 'refresh_seed' not in st.session_state:
    st.session_state.refresh_seed = 0

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 DreamStream")
    if st.session_state.user_mode == 'member':
        st.success(f"Xin chào, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Chức năng", ["Đề xuất AI", "Tìm kiếm Phim", "Theo Thể loại Yêu thích", "Thống kê Cá nhân"])
        if st.button("Đăng xuất"):
            st.session_state.user_mode = None
            st.session_state.current_user = None
            st.rerun()
    elif st.session_state.user_mode in ['guest', 'register']:
        st.info(f"Chế độ: {st.session_state.user_mode.upper()}")
        menu = st.radio("Chức năng", ["Đề xuất AI (Cơ bản)", "Theo Thể loại Đã chọn"])
        if st.button("Thoát chế độ Khách"):
            st.session_state.user_mode = None
            st.session_state.user_genres = []
            st.rerun()
    else:
        menu = "Login"

# --- Main Content ---

if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập Thành viên", "Đăng ký Mới", "Chế độ Khách"])
    with tab1:
        username = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            user_row = users_df[users_df['Tên người dùng'] == username]
            if not user_row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = user_row.iloc[0]
                st.rerun()
    with tab2:
        new_user = st.text_input("Tạo tên người dùng mới")
        selected_g = st.multiselect("Chọn thể loại bạn thích:", ALL_GENRES, key='reg_genres')
        if st.button("Đăng ký & Vào ngay"):
            if new_user and selected_g:
                st.session_state.user_mode = 'register'
                st.session_state.current_user = {'Tên người dùng': new_user}
                st.session_state.user_genres = selected_g
                st.rerun()
    with tab3:
        guest_g = st.multiselect("Chọn thể loại muốn xem:", ALL_GENRES, key='guest_genres')
        if st.button("Truy cập ngay"):
            if guest_g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = guest_g
                st.rerun()

# 2. CHỨC NĂNG DÀNH CHO THÀNH VIÊN
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']
    
    if menu == "Đề xuất AI":
        st.header(f"🤖 Đề xuất Phim Thông minh cho {st.session_state.current_user['Tên người dùng']}")
        
        # Nút Tạo mới đề xuất
        st.markdown('<div class="refresh-button">', unsafe_allow_html=True)
        if st.button("🔄 Tạo mới đề xuất (Đổi 10 phim khác)"):
            st.session_state.refresh_seed += 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        recs = get_ai_recommendations(user_history, seed=st.session_state.refresh_seed)
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(row['Link Poster'], use_container_width=True)
                st.caption(f"**{row['Tên phim']}**")

    elif menu == "Theo Thể loại Yêu thích":
        st.header("❤️ Đề xuất theo Thể loại Yêu thích")
        
        # Nút Tạo mới đề xuất
        if st.button("🔄 Tạo mới đề xuất"):
            st.session_state.refresh_seed += 1
            st.rerun()

        fav_movie = st.session_state.current_user['Phim yêu thích nhất']
        row = movies_df[movies_df['Tên phim'] == fav_movie]
        if not row.empty:
            fav_genres = [x.strip() for x in row.iloc[0]['Thể loại phim'].split(',')]
            recs = get_genre_recommendations(fav_genres, seed=st.session_state.refresh_seed)
            cols = st.columns(5)
            for i, (idx, r) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(r['Link Poster'], use_container_width=True)
                    st.caption(r['Tên phim'])

# 3. CHỨC NĂNG DÀNH CHO KHÁCH / ĐĂNG KÝ MỚI
elif st.session_state.user_mode in ['guest', 'register']:
    selected_g = st.session_state.user_genres
    
    if menu == "Đề xuất AI (Cơ bản)" or menu == "Theo Thể loại Đã chọn":
        st.header("✨ Gợi ý dành cho bạn")
        
        # Nút Tạo mới đề xuất
        if st.button("🔄 Tạo mới danh sách"):
            st.session_state.refresh_seed += 1
            st.rerun()
            
        recs = get_genre_recommendations(selected_g, seed=st.session_state.refresh_seed)
        cols = st.columns(5)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(row['Link Poster'], use_container_width=True)
                st.caption(row['Tên phim'])
