import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
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
    df = pd.read_csv("data_phim_full_images.csv").fillna("")
    return df

@st.cache_data
def load_users():
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv")

movies_df = load_movies()
users_df = load_users()

# ======================================================
# SAFE COLUMN HELPER (CHỐNG KEYERROR)
# ======================================================
def safe_col(df, *cols):
    for c in cols:
        if c in df.columns:
            return df[c].astype(str)
    return ""

# ======================================================
# PREPROCESS CONTENT (KHÔNG CRASH)
# ======================================================
movies_df["content"] = (
    safe_col(movies_df, "Thể loại phim", "Genre") + " " +
    safe_col(movies_df, "Diễn viên chính", "Diễn viên", "Cast", "Actors") + " " +
    safe_col(movies_df, "Đạo diễn", "Director")
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ======================================================
# HELPER: HIỂN THỊ POSTER (FIX ẢNH)
# ======================================================
def get_poster(row):
    for col in ["Link Poster", "Link Backdrop", "poster", "image"]:
        if col in row and str(row[col]).startswith("http"):
            return row[col]
    return "https://via.placeholder.com/300x450?text=No+Image"

def show_movies(df):
    cols = st.columns(5)
    for i, row in enumerate(df.to_dict("records")):
        with cols[i % 5]:
            if st.button(" ", key=f"movie_{row['Tên phim']}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.rerun()
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])

# ======================================================
# RECOMMEND FUNCTIONS
# ======================================================
def content_based(movie_name, top_n=10):
    if movie_name not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)

    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return movies_df.iloc[movie_indices]

def profile_based(user_row, top_n=10):
    watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]

    if genres.empty:
        return movies_df.sample(top_n)

    main_genre = genres.str.split(",").explode().value_counts().idxmax()
    df = movies_df[movies_df["Thể loại phim"].str.contains(main_genre, na=False)]
    df = df[~df["Tên phim"].isin(watched)]

    return df.sample(min(top_n, len(df)))

def genre_based(genres, top_n=10):
    df = movies_df[
        movies_df["Thể loại phim"].apply(
            lambda x: any(g in x for g in genres)
        )
    ]
    return df.sample(min(top_n, len(df)))

# ======================================================
# LOGIN / REGISTER / GUEST (GIỮ UI)
# ======================================================
if st.session_state.logged_in_user is None:
    st.markdown("## 🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    with tab1:
        username = st.selectbox("Tên người dùng:", users_df["Tên người dùng"])
        if st.button("Đăng Nhập"):
            st.session_state.logged_in_user = username
            st.rerun()

    with tab2:
        st.text_input("Tên người dùng bạn muốn tạo:")

    with tab3:
        if st.button("Truy cập với tư cách Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.rerun()

    st.stop()

# ======================================================
# SIDEBAR (GIỮ NGUYÊN)
# ======================================================
st.sidebar.markdown("## Menu Chức Năng")

menu = st.sidebar.radio(
    "Chọn chức năng:",
    [
        "Đề xuất theo Tên Phim",
        "Đề xuất theo AI",
        "Đề xuất theo Thể loại Yêu thích",
        "Đăng Xuất"
    ]
)

if menu == "Đăng Xuất":
    st.session_state.logged_in_user = None
    st.session_state.selected_movie = None
    st.session_state.last_results = None
    st.rerun()

# ======================================================
# LOAD CURRENT USER
# ======================================================
st.markdown(f"## 🎬 Chào mừng, {st.session_state.logged_in_user}!")

if st.session_state.logged_in_user != "GUEST":
    user_row = users_df[
        users_df["Tên người dùng"] == st.session_state.logged_in_user
    ].iloc[0]
else:
    user_row = None

# ======================================================
# MOVIE DETAIL PAGE (THÊM MỚI)
# ======================================================
if st.session_state.selected_movie:
    movie = movies_df[
        movies_df["Tên phim"] == st.session_state.selected_movie
    ].iloc[0]

    st.image(get_poster(movie), use_container_width=True)
    st.title(movie["Tên phim"])

    col1, col2 = st.columns([2, 3])

    with col1:
        st.write("🎭 **Thể loại:**", movie.get("Thể loại phim", ""))
        st.write("🎬 **Đạo diễn:**", movie.get("Đạo diễn", ""))
        st.write("⭐ **Diễn viên:**", movie.get("Diễn viên chính", ""))
        st.write("📅 **Năm:**", movie.get("Năm phát hành", ""))
        st.write("🔥 **Độ phổ biến:**", movie.get("Độ phổ biến", ""))

    with col2:
        st.subheader("🎯 Phim tương tự")
        similar = content_based(movie["Tên phim"], top_n=5)
        show_movies(similar)

    if st.button("⬅️ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()

    st.stop()

# ======================================================
# FEATURE: CONTENT-BASED
# ======================================================
if menu == "Đề xuất theo Tên Phim":
    st.markdown("### 1️⃣ Đề xuất theo Nội dung (Content-Based)")
    movie_name = st.selectbox("Chọn tên phim:", movies_df["Tên phim"])
    if st.button("Tìm Đề Xuất"):
        st.session_state.last_results = content_based(movie_name)

# ======================================================
# FEATURE: PROFILE-BASED AI
# ======================================================
elif menu == "Đề xuất theo AI":
    st.markdown("### 2️⃣ Đề xuất theo AI (Profile-Based)")
    if st.button("Tìm Đề Xuất AI"):
        if user_row is not None:
            st.session_state.last_results = profile_based(user_row)
        else:
            st.session_state.last_results = movies_df.sample(10)

# ======================================================
# FEATURE: GENRE-BASED
# ======================================================
elif menu == "Đề xuất theo Thể loại Yêu thích":
    st.markdown("### 3️⃣ Đề xuất theo Thể loại Yêu thích")
    if user_row is not None:
        genres = movies_df[
            movies_df["Tên phim"].isin(
                ast.literal_eval(user_row["5 phim coi gần nhất"])
            )
        ]["Thể loại phim"].str.split(",").explode().unique().tolist()

        if st.button("Chạy lại Đề xuất AI theo Thể loại này"):
            st.session_state.last_results = genre_based(genres)

# ======================================================
# SHOW RESULTS
# ======================================================
if st.session_state.last_results is not None:
    st.markdown("---")
    show_movies(st.session_state.last_results)
