import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="DreamStream",
    layout="wide"
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
# PREPROCESS
# ======================================================
def safe_col(df, col):
    return df[col].astype(str) if col in df.columns else ""

movies_df["content"] = (
    safe_col(movies_df, "Thể loại phim") + " " +
    safe_col(movies_df, "Diễn viên") + " " +
    safe_col(movies_df, "Đạo diễn")
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "logged_in_user": None,
    "selected_movie": None,
    "last_results": None,
    "guest_genres": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# UTILS
# ======================================================
def get_all_genres():
    s = set()
    for g in movies_df["Thể loại phim"]:
        for x in str(g).split(","):
            s.add(x.strip())
    return sorted(s)

ALL_GENRES = get_all_genres()

def get_poster(row):
    if "Link Poster" in row and str(row["Link Poster"]).startswith("http"):
        return row["Link Poster"]
    return "https://via.placeholder.com/300x450?text=No+Image"

def show_movies(df):
    if df is None or df.empty:
        st.info("Không có phim để hiển thị")
        return
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])
            if st.button("🎬 Xem chi tiết", key=f"{row['Tên phim']}_{i}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.rerun()

# ======================================================
# RECOMMEND FUNCTIONS
# ======================================================
def content_based(title, top_n=10):
    if title not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)
    idx = movies_df[movies_df["Tên phim"] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies_df.iloc[[i[0] for i in scores]]

def recommend_by_genres(genres, top_n=10):
    if not genres:
        return movies_df.sample(top_n)
    df = movies_df[movies_df["Thể loại phim"].apply(
        lambda x: any(g in x for g in genres)
    )]
    return df.sample(min(len(df), top_n)) if not df.empty else movies_df.sample(top_n)

def profile_based(user_row, top_n=10):
    try:
        watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    except:
        watched = []

    watched = watched[:5]
    if not watched:
        return movies_df.sample(top_n)

    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]
    if genres.empty:
        return movies_df.sample(top_n)

    main = genres.str.split(",").explode().value_counts().idxmax()
    return movies_df[movies_df["Thể loại phim"].str.contains(main, na=False)].sample(top_n)

# ======================================================
# 📊 PERSONAL WATCH TREND CHART (QUAN TRỌNG)
# ======================================================
def plot_personal_watch_trend(user_row):
    try:
        watched_movies = ast.literal_eval(user_row["5 phim coi gần nhất"])
    except:
        watched_movies = []

    if not watched_movies:
        st.info("Chưa có lịch sử xem phim để vẽ biểu đồ")
        return

    genres = []
    for movie in watched_movies:
        row = movies_df[movies_df["Tên phim"] == movie]
        if not row.empty:
            genres.extend(row.iloc[0]["Thể loại phim"].split(","))

    if not genres:
        st.info("Không đủ dữ liệu thể loại")
        return

    counter = Counter([g.strip() for g in genres])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counter.keys(), counter.values())
    ax.set_title("📊 Xu hướng xem phim cá nhân")
    ax.set_xlabel("Thể loại")
    ax.set_ylabel("Số lần xem")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ======================================================
# LOGIN / REGISTER / GUEST
# ======================================================
if st.session_state.logged_in_user is None:
    st.title("🍿 DreamStream")
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if u in users_df["Tên người dùng"].values:
                st.session_state.logged_in_user = u
                st.rerun()

    with tab2:
        new = st.text_input("Tên người dùng mới")
        g = st.multiselect("Chọn thể loại yêu thích", ALL_GENRES)
        if st.button("Hoàn tất"):
            if new:
                st.session_state.logged_in_user = new
                st.session_state.guest_genres = g
                st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại muốn xem", ALL_GENRES)
        if st.button("Vào chế độ Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.session_state.guest_genres = g
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
# SIDEBAR MENU
# ======================================================
menu = st.sidebar.radio(
    "Menu",
    [
        "Cá nhân",
        "Đề xuất theo Tên Phim",
        "Đề xuất theo AI",
        "Đề xuất theo Thể loại",
        "Đăng xuất"
    ]
)

if menu == "Đăng xuất":
    st.session_state.clear()
    st.rerun()

st.header(f"🎬 Chào mừng, {st.session_state.logged_in_user}")

# ======================================================
# MENU SCREENS
# ======================================================
if menu == "Cá nhân":
    if st.session_state.logged_in_user == "GUEST":
        st.info("Chế độ Khách không có lịch sử cá nhân")
    else:
        user = users_df[
            users_df["Tên người dùng"] == st.session_state.logged_in_user
        ].iloc[0]

        st.subheader("🎞️ 5 phim đã xem gần nhất")
        try:
            recent_movies = ast.literal_eval(user["5 phim coi gần nhất"])[:5]
        except:
            recent_movies = []

        show_movies(movies_df[movies_df["Tên phim"].isin(recent_movies)])

        st.subheader("📊 Xu hướng xem phim của bạn")
        plot_personal_watch_trend(user)

elif menu == "Đề xuất theo Tên Phim":
    movie = st.selectbox("Chọn phim", movies_df["Tên phim"])
    if st.button("Đề xuất"):
        st.session_state.last_results = content_based(movie)
        show_movies(st.session_state.last_results)

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

    if st.session_state.last_results is not None:
        show_movies(st.session_state.last_results)

elif menu == "Đề xuất theo Thể loại":
    st.session_state.last_results = recommend_by_genres(
        st.session_state.guest_genres
    )
    show_movies(st.session_state.last_results)
