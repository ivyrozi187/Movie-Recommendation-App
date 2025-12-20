import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# 1. CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Movie Recommendation AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation AI")

# ==============================================================================
# 2. LOAD & PREPROCESS DATA
# ==============================================================================
@st.cache_resource
def load_data():
    movies = pd.read_csv("data_phim_full_images.csv")

    # Fill missing
    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Tên phim'] = movies['Tên phim'].fillna('')

    # Combined features
    movies['combined_features'] = (
        movies['Tên phim'] + " " +
        movies['Đạo diễn'] + " " +
        movies['Thể loại phim']
    )

    # Popularity scaling
    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    # TF-IDF + Cosine
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, cosine_sim

movies_df, cosine_sim = load_data()

# ==============================================================================
# 3. SESSION STATE
# ==============================================================================
if 'ai_seen_indices' not in st.session_state:
    st.session_state.ai_seen_indices = set()

# ==============================================================================
# 4. AI RECOMMENDATION FUNCTION (HOÀN CHỈNH)
# ==============================================================================
def get_ai_recommendations(
    history_titles,
    top_k=10,
    w_sim=0.7,
    w_pop=0.3,
    exclude_indices=None
):
    if exclude_indices is None:
        exclude_indices = set()

    # Lấy index phim đã xem
    watched_indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            watched_indices.append(idx[0])

    # Nếu chưa có lịch sử → đề xuất theo độ phổ biến
    if not watched_indices:
        df = movies_df[~movies_df.index.isin(exclude_indices)]
        result = df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)
        return result, list(result.index)

    # Similarity score
    sim_scores = np.mean(cosine_sim[watched_indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values

    # Hybrid score
    final_scores = (w_sim * sim_scores) + (w_pop * pop_scores)

    ranked = sorted(
        enumerate(final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Loại phim đã xem + đã đề xuất
    rec_indices = [
        i for i, _ in ranked
        if i not in watched_indices and i not in exclude_indices
    ][:top_k]

    return movies_df.iloc[rec_indices], rec_indices

# ==============================================================================
# 5. UI
# ==============================================================================

st.subheader("📌 Nhập lịch sử phim đã xem (để test AI)")
history_titles = st.multiselect(
    "Chọn vài phim bạn đã xem:",
    movies_df['Tên phim'].tolist()
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔄 Tạo đề xuất mới"):
        st.session_state.ai_seen_indices.clear()

with col2:
    st.write("Mỗi lần bấm sẽ tạo **10 phim khác – không trùng phim cũ**")

st.divider()

# ==============================================================================
# 6. SHOW RECOMMENDATIONS
# ==============================================================================
recs, new_indices = get_ai_recommendations(
    history_titles,
    exclude_indices=st.session_state.ai_seen_indices
)

st.session_state.ai_seen_indices.update(new_indices)

if recs.empty:
    st.warning("Không còn phim để đề xuất.")
else:
    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.image(row['Link Poster'], use_container_width=True)
            st.caption(row['Tên phim'])
