import streamlit as st
import pandas as pd
import ast
import random
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
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv").fillna("")

movies_df = load_movies()
users_df = load_users()

# ======================================================
# SAFE CONTENT
# ======================================================
def safe_col(df, *cols):
    for c in cols:
        if c in df.columns:
            return df[c].astype(str)
    return ""

movies_df["content"] = (
    safe_col(movies_df, "Thể loại phim") + " " +
    safe_col(movies_df, "Diễn viên", "Diễn viên chính") + " " +
    safe_col(movies_df, "Đạo diễn")
)

# ======================================================
# TF-IDF
# ======================================================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ======================================================
# SESSION
# ======================================================
for k in [
    "logged_in_user",
    "selected_movie",
    "last_results",
    "user_genres",
    "is_new_user",
    "guest_genres"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# ======================================================
# GENRES
# ======================================================
def get_all_genres():
    s = set()
    for g in movies_df["Thể loại phim"]:
        for x in str(g).split(","):
            s.add(x.strip())
    return sorted(s)

ALL_GENRES = get_all_genres()

# ======================================================
# POSTER
# ======================================================
def get_poster(row):
    for c in ["Link Poster", "Link Backdrop"]:
        if c in row and str(row[c]).startswith("http"):
            return row[c]
    return "https://via.placeholder.com/300x450?text=No+Image"

# ======================================================
# SHOW MOVIES
# ======================================================
def show_movies(df):
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])
            if st.button("🎬 Xem chi tiết", key=f"detail_{i}_{row['Tên phim']}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.rerun()

# ======================================================
# RECOMMEND FUNCTIONS
# ======================================================
def content_based(movie_name, top_n=10):
    if movie_name not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)
    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies_df.iloc[[i[0] for i in scores]]

def recommend_by_genres(genres, top_n=10):
    df = movies_df[movies_df["Thể loại phim"].apply(
        lambda x: any(g in x for g in genres)
    )]
    return df.sample(min(top_n, len(df))) if not df.empty else movies_df.sample(top_n)

def profile_based(user_row, top_n=10):
    watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]
    if genres.empty:
        return movies_df.sample(top_n)
    main = genres.str.split(",").explode().value_counts().idxmax()
    df = movies_df[movies_df["Thể loại phim"].str.contains(main, na=False)]
    return df.sample(min(top_n, len(df)))

def recommend_from_favorite_movie(user_row, top_n=10):
    fav = user_row["Phim yêu thích nhất"]
    if fav not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)
    genre = movies_df[movies_df["Tên phim"] == fav]["Thể loại phim"].values[0]
    df = movies_df[movies_df["Thể loại phim"].str.contains(genre.split(",")[0], na=False)]
    return df.sample(min(top_n, len(df)))

# ======================================================
# LOGIN / REGISTER / GUEST
# ======================================================
if st.session_state.logged_in_user is None:
    st.title("🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    # LOGIN
    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if u in users_df["Tên người dùng"].values:
                st.session_state.logged_in_user = u
                st.rerun()
            else:
                st.error("❌ Không tồn tại")

    # REGISTER
    with tab2:
        new = st.text_input("Tên người dùng mới")
        g = st.multiselect("Chọn ≥ 3 thể loại", ALL_GENRES)
        if st.button("Hoàn tất"):
            if new and len(g) >= 3:
                st.session_state.logged_in_user = new
                st.session_state.user_genres = g
                st.session_state.is_new_user = True
                st.rerun()

    # GUEST (CHỌN GENRE → AI)
    with tab3:
        st.session_state.guest_genres = st.multiselect(
            "Chọn thể loại bạn thích:",
            ALL_GENRES
        )
        if st.button("Truy cập với tư cách Khách"):
            if len(st.session_state.guest_genres) >= 2:
                st.session_state.logged_in_user = "GUEST"
                st.rerun()
            else:
                st.warning("⚠️ Chọn ít nhất 2 thể loại")

    st.stop()

# ======================================================
# DETAIL PAGE
# ======================================================
if st.session_state.selected_movie:
    m = movies_df[movies_df["Tên phim"] == st.session_state.selected_movie].iloc[0]
    st.image(get_poster(m), use_container_width=True)
    st.title(m["Tên phim"])
    st.write("🎭", m["Thể loại phim"])
    st.subheader("🎯 Phim tương tự")
    show_movies(content_based(m["Tên phim"], 5))
    if st.button("⬅️ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()
    st.stop()

# ======================================================
# SIDEBAR
# ======================================================
menu = st.sidebar.radio(
    "Menu",
    [
        "Đề xuất theo Tên Phim",
        "Đề xuất theo AI",
        "Đề xuất theo Thể loại Yêu thích",
        "Đăng Xuất"
    ]
)

if menu == "Đăng Xuất":
    st.session_state.clear()
    st.rerun()

# ======================================================
# HOME
# ======================================================
st.header(f"🎬 Chào mừng, {st.session_state.logged_in_user}")

# NEW USER
if st.session_state.is_new_user:
    st.subheader("🌟 Gợi ý ban đầu cho bạn")
    show_movies(recommend_by_genres(st.session_state.user_genres))
    st.session_state.is_new_user = False

# CONTENT BASED
elif menu == "Đề xuất theo Tên Phim":
    movie = st.selectbox("Chọn phim:", movies_df["Tên phim"])
    if st.button("Tìm"):
        st.session_state.last_results = content_based(movie)

# AI
elif menu == "Đề xuất theo AI":
    if st.session_state.logged_in_user == "GUEST":
        st.session_state.last_results = recommend_by_genres(
            st.session_state.guest_genres
        )
    else:
        user = users_df[users_df["Tên người dùng"] == st.session_state.logged_in_user].iloc[0]
        st.session_state.last_results = profile_based(user)

# FAVORITE GENRE + REFRESH
elif menu == "Đề xuất theo Thể loại Yêu thích":
    user = users_df[users_df["Tên người dùng"] == st.session_state.logged_in_user].iloc[0]
    st.session_state.last_results = recommend_from_favorite_movie(user)

    if st.button("🔄 Tạo đề xuất mới"):
        st.session_state.last_results = recommend_from_favorite_movie(user)

# ======================================================
# SHOW
# ======================================================
if st.session_state.last_results is not None:
    st.markdown("---")
    show_movies(st.session_state.last_results)
