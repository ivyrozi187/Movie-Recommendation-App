import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="DreamStream",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SESSION STATE INIT
# =========================
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if "mode" not in st.session_state:
    st.session_state.mode = "login"

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_movies():
    df = pd.read_csv("data_phim_full_images.csv")
    df.fillna("", inplace=True)
    return df

@st.cache_data
def load_users():
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv")

movies_df = load_movies()
users_df = load_users()

# =========================
# PREPROCESS CONTENT
# =========================
movies_df["content"] = (
    movies_df["Thể loại phim"].astype(str) + " " +
    movies_df["Diễn viên"].astype(str) + " " +
    movies_df["Đạo diễn"].astype(str)
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# =========================
# HELPER FUNCTIONS
# =========================
def content_based_recommend(movie_name, top_n=10):
    if movie_name not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)

    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices]

def profile_based_recommend(user_row, top_n=10):
    watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]

    if genres.empty:
        return movies_df.sample(top_n)

    main_genre = genres.str.split(",").explode().value_counts().idxmax()
    df = movies_df[movies_df["Thể loại phim"].str.contains(main_genre, na=False)]
    df = df[~df["Tên phim"].isin(watched)]

    return df.sample(min(top_n, len(df)))

def genre_based_recommend(genres, top_n=10):
    mask = movies_df["Thể loại phim"].apply(
        lambda x: any(g in x for g in genres)
    )
    df = movies_df[mask]
    return df.sample(min(top_n, len(df)))

def show_movies(df):
    cols = st.columns(5)
    for i, row in enumerate(df.itertuples()):
        with cols[i % 5]:
            st.image(row._asdict().get("Link Poster", ""), use_container_width=True)
            st.caption(row._asdict().get("Tên phim", ""))

# =========================
# LOGIN / REGISTER / GUEST UI (GIỮ NGUYÊN LOGIC TAB)
# =========================
if st.session_state.logged_in_user is None:
    st.markdown("## 🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    # -------- LOGIN --------
    with tab1:
        username = st.selectbox(
            "Tên người dùng:",
            users_df["Tên người dùng"]
        )
        if st.button("Đăng Nhập"):
            st.session_state.logged_in_user = username
            st.session_state.mode = "home"

    # -------- REGISTER (UI giữ, demo không ghi file) --------
    with tab2:
        st.text_input("Tên người dùng bạn muốn tạo:")

    # -------- GUEST --------
    with tab3:
        if st.button("Truy Cập với tư cách Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.session_state.mode = "guest"

    st.stop()

# =========================
# SIDEBAR (GIỮ NGUYÊN)
# =========================
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
    st.session_state.last_results = None
    st.stop()

# =========================
# LOAD CURRENT USER
# =========================
if st.session_state.logged_in_user != "GUEST":
    user_row = users_df[
        users_df["Tên người dùng"] == st.session_state.logged_in_user
    ].iloc[0]
else:
    user_row = None

st.markdown(f"## 🎬 Chào mừng, {st.session_state.logged_in_user}!")

# =========================
# FEATURE 1: CONTENT-BASED
# =========================
if menu == "Đề xuất theo Tên Phim":
    st.markdown("### 1️⃣ Đề xuất theo Nội dung (Content-Based)")
    movie_name = st.selectbox("Chọn tên phim:", movies_df["Tên phim"])
    if st.button("Tìm Đề Xuất"):
        st.session_state.last_results = content_based_recommend(movie_name)

# =========================
# FEATURE 2: PROFILE-BASED AI
# =========================
elif menu == "Đề xuất theo AI":
    st.markdown("### 2️⃣ Đề xuất theo AI (Profile-Based)")
    if st.button("Tìm Đề Xuất AI"):
        if user_row is not None:
            st.session_state.last_results = profile_based_recommend(user_row)
        else:
            st.session_state.last_results = movies_df.sample(10)

# =========================
# FEATURE 3: GENRE-BASED
# =========================
elif menu == "Đề xuất theo Thể loại Yêu thích":
    st.markdown("### 3️⃣ Đề xuất theo Thể loại Yêu thích")
    if user_row is not None:
        genres = movies_df[
            movies_df["Tên phim"].isin(
                ast.literal_eval(user_row["5 phim coi gần nhất"])
            )
        ]["Thể loại phim"].str.split(",").explode().unique().tolist()

        if st.button("Chạy lại Đề xuất AI theo Thể loại này"):
            st.session_state.last_results = genre_based_recommend(genres)

# =========================
# SHOW RESULTS
# =========================
if st.session_state.last_results is not None:
    st.markdown("---")
    show_movies(st.session_state.last_results)
