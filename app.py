import streamlit as st
import pandas as pd
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
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv").fillna("")

movies_df = load_movies()
users_df = load_users()

# ======================================================
# SAFE COLUMN
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
for key in [
    "logged_in_user",
    "selected_movie",
    "last_results",
    "user_genres",
    "is_new_user"
]:
    if key not in st.session_state:
        st.session_state[key] = None

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
# SHOW MOVIES
# ======================================================
def show_movies(df):
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])

            btn_key = f"detail_{i}_{row['Tên phim']}"
            if st.button("🎬 Xem chi tiết", key=btn_key):
                st.session_state.selected_movie = row["Tên phim"]
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

def profile_based(user_row, top_n=10):
    watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]

    if genres.empty:
        return movies_df.sample(top_n)

    main_genre = genres.str.split(",").explode().value_counts().idxmax()
    df = movies_df[movies_df["Thể loại phim"].str.contains(main_genre, na=False)]
    return df.sample(min(top_n, len(df)))

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
    movie = movies_df[
        movies_df["Tên phim"] == st.session_state.selected_movie
    ].iloc[0]

    st.image(get_poster(movie), use_container_width=True)
    st.title(movie["Tên phim"])

    st.write("🎭 **Thể loại:**", movie.get("Thể loại phim", ""))
    st.write("🎬 **Đạo diễn:**", movie.get("Đạo diễn", ""))
    st.write("⭐ **Diễn viên:**", movie.get("Diễn viên chính", ""))

    st.subheader("🎯 Phim tương tự")
    show_movies(content_based(movie["Tên phim"], 5))

    if st.button("⬅️ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()

    st.stop()

# ======================================================
# SIDEBAR
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
# HOME
# ======================================================
st.markdown(f"## 🎬 Chào mừng, {st.session_state.logged_in_user}")

if st.session_state.is_new_user:
    st.subheader("🌟 Gợi ý cho bạn (Dựa trên thể loại yêu thích)")
    show_movies(recommend_by_genres(st.session_state.user_genres))
    st.session_state.is_new_user = False

elif menu == "Đề xuất theo Tên Phim":
    movie = st.selectbox("Chọn phim:", movies_df["Tên phim"])
    if st.button("Tìm Đề Xuất"):
        st.session_state.last_results = content_based(movie)

elif menu == "Đề xuất theo AI":
    user_row = users_df[
        users_df["Tên người dùng"] == st.session_state.logged_in_user
    ].iloc[0]
    if st.button("Tìm Đề Xuất AI"):
        st.session_state.last_results = profile_based(user_row)

elif menu == "Đề xuất theo Thể loại Yêu thích":
    if st.session_state.user_genres:
        st.session_state.last_results = recommend_by_genres(
            st.session_state.user_genres
        )
    else:
        st.info("⚠️ Người dùng chưa có thể loại yêu thích")

# ======================================================
# SHOW RESULTS
# ======================================================
if st.session_state.last_results is not None:
    st.markdown("---")
    show_movies(st.session_state.last_results)
