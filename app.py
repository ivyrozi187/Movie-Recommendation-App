import streamlit as st
import pandas as pd
import ast
import random
import matplotlib.pyplot as plt
from collections import Counter
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
# SESSION STATE
# ======================================================
for k in [
    "logged_in_user",
    "selected_movie",
    "last_results",
    "user_genres",
    "guest_genres",
    "is_new_user"
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
    df = movies_df[
        movies_df["Thể loại phim"].apply(
            lambda x: any(g in x for g in genres)
        )
    ]
    return df.sample(min(top_n, len(df))) if not df.empty else movies_df.sample(top_n)

def profile_based(user_row, top_n=10):
    try:
        watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    except:
        watched = []

    watched = watched[:5]

    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]
    if genres.empty:
        return movies_df.sample(top_n)
    main = genres.str.split(",").explode().value_counts().idxmax()
    df = movies_df[movies_df["Thể loại phim"].str.contains(main, na=False)]
    return df.sample(min(top_n, len(df)))

# ======================================================
# 📊 USER TREND CHART
# ======================================================
def plot_user_trend_from_movies(movie_list):
    genres = []
    for m in movie_list:
        row = movies_df[movies_df["Tên phim"] == m]
        if not row.empty:
            genres.extend(row.iloc[0]["Thể loại phim"].split(","))

    if not genres:
        st.info("Không đủ dữ liệu để vẽ biểu đồ xu hướng")
        return

    counter = Counter([g.strip() for g in genres])
    labels = list(counter.keys())
    values = list(counter.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#f4a7b9")
    ax.set_title("Xu hướng xem phim của bạn")
    ax.set_ylabel("Số lần")
    ax.set_xlabel("Thể loại")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ======================================================
# LOGIN / REGISTER / GUEST
# ======================================================
if st.session_state.logged_in_user is None:
    st.title("🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if u in users_df["Tên người dùng"].values:
                st.session_state.logged_in_user = u
                st.rerun()

    with tab2:
        new = st.text_input("Tên người dùng mới")
        g = st.multiselect("Chọn thể loại bạn thích:", ALL_GENRES)
        if st.button("Hoàn tất & Đề xuất"):
            if new and len(g) >= 2:
                st.session_state.logged_in_user = new
                st.session_state.user_genres = g
                st.session_state.is_new_user = True
                st.rerun()

    with tab3:
        st.session_state.guest_genres = st.multiselect(
            "Chọn thể loại muốn xem:",
            ALL_GENRES
        )
        if st.button("Truy cập với tư cách Khách"):
            if len(st.session_state.guest_genres) >= 1:
                st.session_state.logged_in_user = "GUEST"
                st.rerun()

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
        "Cá nhân",
        "Tìm phim theo tên",
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

# ================== 👤 CÁ NHÂN ==================
if menu == "Cá nhân":
    if st.session_state.logged_in_user == "GUEST":
        st.info("Chế độ Khách không có trang Cá nhân")
    else:
        user = users_df[
            users_df["Tên người dùng"] == st.session_state.logged_in_user
        ].iloc[0]

        st.subheader("🎞️ 5 phim bạn xem gần nhất")

        try:
            recent_movies = ast.literal_eval(user["5 phim coi gần nhất"])
        except:
            recent_movies = []

        recent_movies = recent_movies[:5]

        recent_df = movies_df[movies_df["Tên phim"].isin(recent_movies)]
        show_movies(recent_df)

        st.subheader("📊 Xu hướng xem phim")
        plot_user_trend_from_movies(recent_movies)

# ================== 🔍 TÌM PHIM THEO TÊN ==================
elif menu == "Tìm phim theo tên":
    keyword = st.text_input("Nhập tên phim cần tìm")

    if keyword:
        result = movies_df[
            movies_df["Tên phim"].str.lower() == keyword.lower()
        ]

        if not result.empty:
            show_movies(result)
        else:
            st.warning("❌ Không tìm thấy phim")

# ================== 🤖 AI ==================
elif menu == "Đề xuất theo AI":
    if st.button("🎬 Đề xuất AI"):
        if st.session_state.logged_in_user == "GUEST":
            st.session_state.last_results = recommend_by_genres(
                st.session_state.guest_genres
            )
        else:
            user = users_df[
                users_df["Tên người dùng"] == st.session_state.logged_in_user
            ].iloc[0]
            st.session_state.last_results = profile_based(user)

    if st.button("🔄 Tạo đề xuất mới"):
        if st.session_state.logged_in_user == "GUEST":
            st.session_state.last_results = recommend_by_genres(
                st.session_state.guest_genres
            )
        else:
            user = users_df[
                users_df["Tên người dùng"] == st.session_state.logged_in_user
            ].iloc[0]
            st.session_state.last_results = profile_based(user)

# ================== 🎯 GENRE ==================
elif menu == "Đề xuất theo Thể loại Yêu thích":
    if st.session_state.logged_in_user == "GUEST":
        st.session_state.last_results = recommend_by_genres(
            st.session_state.guest_genres
        )
        if st.button("🔄 Tạo đề xuất mới"):
            st.session_state.last_results = recommend_by_genres(
                st.session_state.guest_genres
            )
    else:
        user = users_df[
            users_df["Tên người dùng"] == st.session_state.logged_in_user
        ].iloc[0]
        fav = user["Phim yêu thích nhất"]
        if fav in movies_df["Tên phim"].values:
            g = movies_df[
                movies_df["Tên phim"] == fav
            ]["Thể loại phim"].values[0].split(",")

            st.session_state.last_results = recommend_by_genres(g)

            if st.button("🔄 Tạo đề xuất mới"):
                st.session_state.last_results = recommend_by_genres(g)

# ======================================================
# SHOW RESULTS
# ======================================================
if st.session_state.last_results is not None and menu != "Cá nhân":
    st.markdown("---")
    show_movies(st.session_state.last_results)
