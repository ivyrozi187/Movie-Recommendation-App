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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD & XỬ LÝ DỮ LIỆU
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
        for x in g.split(','):
            all_genres.add(x.strip())

    return movies, users, cosine_sim, sorted(list(all_genres))

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. SESSION STATE – PHỤC VỤ NÚT "TẠO MỚI"
# ==============================================================================
if 'ai_seen' not in st.session_state:
    st.session_state.ai_seen = set()

if 'search_seen' not in st.session_state:
    st.session_state.search_seen = set()

if 'genre_seen' not in st.session_state:
    st.session_state.genre_seen = set()

# ==============================================================================
# 4. CÁC HÀM ĐỀ XUẤT
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, exclude=None):
    if exclude is None:
        exclude = set()

    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])

    sim_scores = np.mean(cosine_sim[indices], axis=0) if indices else np.zeros(len(movies_df))
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)

    scores = list(enumerate(final_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rec_idx = [
        i for i, _ in scores
        if i not in indices and i not in exclude
    ][:top_k]

    return movies_df.iloc[rec_idx], rec_idx


def get_genre_recommendations(genres, top_k=10, exclude=None):
    if exclude is None:
        exclude = set()

    pattern = "|".join(genres)
    df = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    df = df[~df.index.isin(exclude)]

    result = df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)
    return result, list(result.index)

# ==============================================================================
# 5. GIAO DIỆN – DEMO 3 CHỨC NĂNG
# ==============================================================================

st.header("🤖 ĐỀ XUẤT AI (Similarity + Popularity)")

if st.button("🔄 Tạo mới đề xuất AI"):
    st.session_state.ai_seen = set()

recs, idxs = get_ai_recommendations(
    users_df.iloc[0]['history_list'],
    exclude=st.session_state.ai_seen
)
st.session_state.ai_seen.update(idxs)

cols = st.columns(5)
for i, (_, r) in enumerate(recs.iterrows()):
    with cols[i % 5]:
        st.image(r['Link Poster'], use_container_width=True)
        st.caption(r['Tên phim'])

st.divider()

# ------------------------------------------------------------------------------
st.header("🔍 TÌM KIẾM & PHIM TƯƠNG TỰ")

movie_name = st.text_input("Nhập tên phim:")

if movie_name:
    result = movies_df[movies_df['Tên phim'].str.contains(movie_name, case=False)]

    if not result.empty:
        movie = result.iloc[0]

        st.subheader("🎬 Phim tương tự")

        if st.button("🔄 Tạo mới phim tương tự"):
            st.session_state.search_seen = set()

        recs, idxs = get_ai_recommendations(
            [movie['Tên phim']],
            w_sim=1.0,
            w_pop=0.0,
            exclude=st.session_state.search_seen
        )
        st.session_state.search_seen.update(idxs)

        cols = st.columns(5)
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i]:
                st.image(r['Link Poster'], use_container_width=True)
                st.caption(r['Tên phim'])

st.divider()

# ------------------------------------------------------------------------------
st.header("🎭 ĐỀ XUẤT THEO THỂ LOẠI")

genres = st.multiselect("Chọn thể loại:", ALL_GENRES)

if genres:
    if st.button("🔄 Tạo mới theo thể loại"):
        st.session_state.genre_seen = set()

    recs, idxs = get_genre_recommendations(
        genres,
        exclude=st.session_state.genre_seen
    )
    st.session_state.genre_seen.update(idxs)

    cols = st.columns(5)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.image(r['Link Poster'], use_container_width=True)
            st.caption(r['Tên phim'])
