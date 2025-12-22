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
# 1. CẤU HÌNH TRANG
# ==============================================================================
st.set_page_config(
    page_title="DreamStream - Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CSS
# ==============================================================================
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOAD & XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")

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

    users['history_list'] = users['5 phim coi gần nhất'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    all_genres = set()
    for g in movies['Thể loại phim']:
        for i in g.split(','):
            all_genres.add(i.strip())

    return movies, users, cosine_sim, sorted(list(all_genres))

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 4. SESSION STATE (QUAN TRỌNG)
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

# 👉 LƯU PHIM ĐÃ ĐỀ XUẤT (KHÔNG TRÙNG)
if 'shown_ai_recs' not in st.session_state:
    st.session_state.shown_ai_recs = set()
if 'shown_genre_recs' not in st.session_state:
    st.session_state.shown_genre_recs = set()

# ==============================================================================
# 5. HÀM CỐT LÕI
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=50, w_sim=0.7, w_pop=0.3):
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])

    if not indices:
        return movies_df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = w_sim * sim_scores + w_pop * pop_scores

    scores = list(enumerate(final_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in scores if i[0] not in indices][:top_k]

    return movies_df.iloc[rec_indices]

def get_genre_recommendations(genres, top_k=50):
    if not genres:
        return pd.DataFrame()
    pattern = '|'.join(genres)
    df = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    return df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

# 👉 HÀM LẤY 10 PHIM KHÔNG TRÙNG
def get_new_recommendations(df, shown_set, top_k=10):
    remain = df[~df['Tên phim'].isin(shown_set)]
    if remain.empty:
        shown_set.clear()
        remain = df
    result = remain.head(top_k)
    shown_set.update(result['Tên phim'].tolist())
    return result

# ==============================================================================
# 6. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎬 DreamStream")
    st.write("Hệ thống gợi ý phim thông minh")

    if st.session_state.user_mode == 'member':
        st.success(f"Xin chào, {st.session_state.current_user['Tên người dùng']}!")
        menu = st.radio("Chức năng", [
            "Đề xuất AI",
            "Theo Thể loại Yêu thích"
        ])
        if st.button("Đăng xuất"):
            st.session_state.clear()
            st.rerun()

    elif st.session_state.user_mode in ['guest', 'register']:
        st.info(f"Chế độ: {st.session_state.user_mode.upper()}")
        menu = st.radio("Chức năng", [
            "Đề xuất AI",
            "Theo Thể loại"
        ])
        if st.button("Thoát"):
            st.session_state.clear()
            st.rerun()

    else:
        menu = "Login"

# ==============================================================================
# 7. LOGIN / REGISTER / GUEST
# ==============================================================================
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        user = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            row = users_df[users_df['Tên người dùng'] == user]
            if not row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = row.iloc[0]
                st.rerun()
            else:
                st.error("Không tồn tại")

    with tab2:
        name = st.text_input("Tên mới")
        g = st.multiselect("Thể loại yêu thích", ALL_GENRES)
        if st.button("Đăng ký"):
            if name and g:
                st.session_state.user_mode = 'register'
                st.session_state.current_user = {'Tên người dùng': name}
                st.session_state.user_genres = g
                st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại", ALL_GENRES)
        if st.button("Vào ngay"):
            if g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = g
                st.rerun()

# ==============================================================================
# 8. MEMBER
# ==============================================================================
elif st.session_state.user_mode == 'member':
    history = st.session_state.current_user['history_list']

    if menu == "Đề xuất AI":
        st.header("🤖 Đề xuất AI")

        if st.button("🔄 Tạo mới"):
            st.session_state.shown_ai_recs.clear()

        base = get_ai_recommendations(history)
        recs = get_new_recommendations(base, st.session_state.shown_ai_recs)

    elif menu == "Theo Thể loại Yêu thích":
        fav = st.session_state.current_user['Phim yêu thích nhất']
        row = movies_df[movies_df['Tên phim'] == fav]
        genres = row.iloc[0]['Thể loại phim'].split(',') if not row.empty else []

        if st.button("🔄 Tạo mới"):
            st.session_state.shown_genre_recs.clear()

        base = get_genre_recommendations(genres)
        recs = get_new_recommendations(base, st.session_state.shown_genre_recs)

    cols = st.columns(5)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.image(r['Link Poster'], use_container_width=True)
            st.caption(r['Tên phim'])

# ==============================================================================
# 9. GUEST / REGISTER
# ==============================================================================
elif st.session_state.user_mode in ['guest', 'register']:
    genres = st.session_state.user_genres

    if st.button("🔄 Tạo mới"):
        st.session_state.shown_ai_recs.clear()

    base = get_genre_recommendations(genres)
    recs = get_new_recommendations(base, st.session_state.shown_ai_recs)

    cols = st.columns(5)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.image(r['Link Poster'], use_container_width=True)
            st.caption(r['Tên phim'])
