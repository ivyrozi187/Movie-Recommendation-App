import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="DreamStream",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_movies():
    return pd.read_csv("data_phim_full_images.csv").fillna("")

@st.cache_data
def load_users():
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv")

movies_df = load_movies()
users_df = load_users()

# ======================================================
# SAFE COLUMN (CHỐNG KEYERROR)
# ======================================================
def safe_col(df, *cols):
    for c in cols:
        if c in df.columns:
            return df[c].astype(str)
    return ""

movies_df["content"] = (
    safe_col(movies_df, "Thể loại phim", "Genre") + " " +
    safe_col(movies_df, "Diễn viên chính", "Diễn viên", "Cast", "Actors") + " " +
    safe_col(movies_df, "Đạo diễn", "Director")
)

# ======================================================
# TF-IDF
# ======================================================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ======================================================
# SESSION STATE
# ======================================================
for k in [
    "logged_in_user", "selected_movie",
    "last_results", "user_genres", "is_new_user"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# ======================================================
# GENRES
# ======================================================
def get_all_genres():
    genres = set()
    for g in movies_df["Thể loại phim"]:
        for x in str(g).split(","):
            genres.add(x.strip())
    return sorted(genres)

ALL_GENRES = get_all_genres()

# ======================================================
# POSTER
# ======================================================
def get_poster(row):
    for col in ["Link Poster", "Link Backdrop", "poster", "image"]:
        if col in row and str(row[col]).startswith("http"):
            return row[col]
    return "https://via.placeholder.com/300x450?text=No+Image"

# ======================================================
# AI EXPLAIN
# ======================================================
def explain_recommendation(movie, user_genres):
    reasons = []
    movie_genres = movie["Thể loại phim"].split(",")

    common = set(g.strip() for g in movie_genres) & set(user_genres)
    if common:
        reasons.append(f"🎭 Cùng thể loại: {', '.join(list(common)[:2])}")

    if movie.get("Đạo diễn"):
        reasons.append(f"🎬 Đạo diễn: {movie['Đạo diễn']}")

    if movie.get("Diễn viên chính"):
        reasons.append(f"⭐ Diễn viên: {movie['Diễn viên chính'].split(',')[0]}")

    return " • ".join(reasons) if reasons else "🔥 Phim phổ biến"

# ======================================================
# SHOW MOVIES (CÓ BUTTON XEM CHI TIẾT)
# ======================================================
def show_movies(df):
    cols = st.columns(5)
    for i, movie in enumerate(df.to_dict("records")):
        with cols[i % 5]:
            st.image(get_poster(movie), use_container_width=True)
            st.caption(movie["Tên phim"])

            if st.session_state.user_genres:
                st.caption(
                    explain_recommendation(movie, st.session_state.user_genres)
                )

            if st.button("🎬 Xem chi tiết", key=f"detail_{movie['Tên phim']}"):
                st.session_state.selected_movie = movie["Tên phim"]
                st.rerun()

# ======================================================
# RECOMMENDERS
# ======================================================
def content_based(movie_name, top_n=10):
    if movie_name not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)

    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies_df.iloc[[i[0] for i in scores]]

def recommend_by_genres(genres, top_n=10):
    mask = movies_df["Thể loại phim"].apply(
        lambda x: any(g in x for g in genres)
    )
    df = movies_df[mask]
    return df.sample(min(top_n, len(df))) if not df.empty else movies_df.sample(top_n)

# ======================================================
# LOGIN / REGISTER / GUEST
# ======================================================
if st.session_state.logged_in_user is None:
    st.markdown("## 🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    # LOGIN
    with tab1:
        username = st.text_input("Tên người dùng:")
        if st.button("Đăng Nhập"):
            if username in users_df["Tên người dùng"].values:
                st.session_state.logged_in_user = username
                st.session_state.user_genres = []
                st.rerun()
            else:
                st.error("❌ Người dùng không tồn tại")

    # REGISTER
    with tab2:
        new_user = st.text_input("Tên người dùng mới:")
        genres = st.multiselect("Chọn ít nhất 3 thể loại:", ALL_GENRES)

        if st.button("Hoàn tất & Xem đề xuất"):
            if new_user and len(genres) >= 3:
                st.session_state.logged_in_user = new_user
                st.session_state.user_genres = genres
                st.session_state.is_new_user = True
                st.rerun()
            else:
                st.warning("⚠️ Nhập tên và chọn ≥ 3 thể loại")

    # GUEST
    with tab3:
        if st.button("Truy cập với tư cách Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.session_state.user_genres = []
            st.rerun()

    st.stop()

# ======================================================
# MOVIE DETAIL PAGE
# ======================================================
if st.session_state.selected_movie:
    movie = movies_df[movies_df["Tên phim"] == st.session_state.selected_movie].iloc[0]

    st.image(get_poster(movie), use_container_width=True)
    st.title(movie["Tên phim"])

    st.write("🎭 **Thể loại:**", movie["Thể loại phim"])
    st.write("🎬 **Đạo diễn:**", movie.get("Đạo diễn", ""))
    st.write("⭐ **Diễn viên:**", movie.get("Diễn viên chính", ""))

    st.subheader("🎯 Phim tương tự")
    show_movies(content_based(movie["Tên phim"], 5))

    if st.button("⬅️ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()

    st.stop()

# ======================================================
# HOME
# ======================================================
st.markdown(f"## 🎬 Chào mừng, {st.session_state.logged_in_user}")

if st.session_state.is_new_user:
    st.subheader("🌟 Gợi ý cho bạn")
    show_movies(recommend_by_genres(st.session_state.user_genres))
    st.session_state.is_new_user = False
else:
    st.subheader("🔥 Phim nổi bật")
    show_movies(movies_df.sample(10))
