import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Netflix Movie Recommender",
    layout="wide"
)

# ===================== LOAD DATA =====================
movies_df = pd.read_csv("movie_info_1000.csv")
users_df = pd.read_csv("user_dataset_with_posters.csv")

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
body {
    background-color: #141414;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.poster img {
    border-radius: 12px;
}
[data-testid="stImage"] {
    transition: transform .2s;
}
[data-testid="stImage"]:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.title("🎥 Netflix Recommender")
user_id = st.sidebar.selectbox(
    "Chọn người dùng",
    users_df["user_id"]
)

user = users_df[users_df["user_id"] == user_id].iloc[0]

# ===================== HEADER =====================
st.title("🍿 Netflix-style Movie Recommendation")
st.subheader(f"Xin chào, **{user['username']}** 👋")

# ===================== RECENT WATCHED =====================
st.markdown("## 🎬 Phim bạn xem gần nhất")

recent_movies = user["recent_movies"].split("|")
recent_posters = user["recent_posters"].split("|")

cols = st.columns(5)
for col, title, poster in zip(cols, recent_movies, recent_posters):
    with col:
        st.image(poster, use_container_width=True)
        st.caption(title)

# ===================== FAVORITE =====================
st.markdown("## ❤️ Phim yêu thích nhất")
st.image(user["favorite_poster"], width=260)
st.write(f"**{user['favorite_movie']}**")

# ===================== SIMPLE CONTENT-BASED RECOMMENDATION =====================
st.markdown("## ⭐ Gợi ý dành cho bạn")

# Lấy thể loại phim yêu thích
fav_movie = user["favorite_movie"]
fav_genre = movies_df[movies_df["Tên phim"] == fav_movie]["Thể loại phim"]

if not fav_genre.empty:
    fav_genre = fav_genre.values[0]
    recommended = movies_df[
        movies_df["Thể loại phim"].str.contains(fav_genre.split(",")[0], na=False)
    ].sample(5)
else:
    recommended = movies_df.sample(5)

rec_cols = st.columns(5)

for col, (_, row) in zip(rec_cols, recommended.iterrows()):
    with col:
        st.image(
            f"https://image.tmdb.org/t/p/w500",  # placeholder nếu bạn muốn fetch thêm
            use_container_width=True
        )
        st.caption(row["Tên phim"])

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("🎓 **BTL – Hệ thống gợi ý phim | Streamlit + TMDb API**")

