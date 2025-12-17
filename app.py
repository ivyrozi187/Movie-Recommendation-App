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
# SHOW MOVIES (FIX BUTTON KEY)
# ======================================================
def show_movies(df):
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])

            if st.session_state.user_genres:
                st.caption(
                    explain_recommendation(row, st.session_state.user_genres)
                )

            btn_key = f"detail_{i}_{row['Tên phim']}"
            if st.button("🎬 Xem chi tiết", key=btn_key):
                st.session_state.selected_movie = row["Tên phim"]
                st.experimental_rerun()

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
                st.experimental_rerun()
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
                st.experimental_rerun()
            else:
                st.warning("⚠️ Nhập tên và chọn ≥ 3 thể loại")

    # GUEST
    with tab3:
        if st.button("Truy cập với tư cách Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.session_state.user_genres = []
            st.experimental_rerun()

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
        st.experimental_rerun()

    st.stop()

# ======================================================
# HOME
# ======================================================
st.markdown(f"## 🎬 Chào mừng, {st.session_state.logged_in_user}")

if st.session_state.is_new_user:
    st.subheader("🌟 Gợi ý cho bạn (Dựa trên thể loại yêu thích)")
    show_movies(recommend_by_genres(st.session_state.user_genres))
    st.session_state.is_new_user = False
else:
    st.subheader("🔥 Phim nổi bật")
    show_movies(movies_df.sample(10))
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
st.set_page_config(
    page_title="DreamStream",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= LOAD DATA =================
@st.cache_data
def load_movies():
    return pd.read_csv("data_phim_full_images.csv").fillna("")

@st.cache_data
def load_users():
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv").fillna("")

movies_df = load_movies()
users_df = load_users()

# ================= SAFE COLUMN =================
def col(name):
    return movies_df[name].astype(str) if name in movies_df.columns else ""

movies_df["content"] = (
    col("Thể loại phim") + " " +
    col("Diễn viên chính") + " " +
    col("Đạo diễn")
)

# ================= TF-IDF =================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ================= SESSION =================
for k in ["user", "mode", "genres", "detail_movie"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ================= POSTER =================
def poster(row):
    return row["Link Poster"] if row["Link Poster"].startswith("http") else "https://via.placeholder.com/300x450"

# ================= RECOMMEND =================
def content_based(movie_name, top=10):
    if movie_name not in movies_df["Tên phim"].values:
        return movies_df.sample(top)
    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top+1]
    return movies_df.iloc[[i[0] for i in scores]]

def genre_based(genres, top=10):
    df = movies_df[movies_df["Thể loại phim"].str.contains("|".join(genres), case=False)]
    return df.sample(min(top, len(df))) if not df.empty else movies_df.sample(top)

# ================= SHOW MOVIES =================
def show_movies(df):
    cols = st.columns(5)
    for i, row in df.iterrows():
        with cols[i % 5]:
            st.image(poster(row), use_container_width=True)
            st.caption(row["Tên phim"])
            if st.button("🎬 Xem chi tiết", key=row["Tên phim"]):
                st.session_state.detail_movie = row["Tên phim"]
                st.rerun()

# ================= LOGIN =================
if st.session_state.user is None:
    st.title("🍿 DreamStream: Đề xuất Phim Cá nhân")
    tab1, tab2, tab3 = st.tabs(["Đăng Nhập", "Đăng Ký", "Chế Độ Khách"])

    with tab1:
        name = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if name in users_df["Tên người dùng"].values:
                st.session_state.user = name
                st.session_state.genres = []
                st.rerun()
            else:
                st.error("❌ Người dùng không tồn tại")

    with tab2:
        new_name = st.text_input("Tên người dùng mới")
        all_genres = sorted(set(",".join(movies_df["Thể loại phim"]).split(",")))
        genres = st.multiselect("Chọn ít nhất 3 thể loại", all_genres)
        if st.button("Hoàn tất & đề xuất"):
            if new_name and len(genres) >= 3:
                st.session_state.user = new_name
                st.session_state.genres = genres
                st.session_state.mode = "genre"
                st.rerun()
            else:
                st.warning("⚠️ Nhập tên + chọn ≥ 3 thể loại")

    with tab3:
        if st.button("Truy cập với tư cách Khách"):
            st.session_state.user = "GUEST"
            st.session_state.genres = []
            st.rerun()

    st.stop()

# ================= DETAIL PAGE =================
if st.session_state.detail_movie:
    m = movies_df[movies_df["Tên phim"] == st.session_state.detail_movie].iloc[0]
    st.image(poster(m), width=300)
    st.title(m["Tên phim"])
    st.write("🎭 Thể loại:", m["Thể loại phim"])
    st.write("🎬 Đạo diễn:", m["Đạo diễn"])
    st.write("⭐ Diễn viên:", m["Diễn viên chính"])

    st.subheader("🎯 Phim tương tự")
    show_movies(content_based(m["Tên phim"], 5))

    if st.button("⬅ Quay lại"):
        st.session_state.detail_movie = None
        st.rerun()

    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("Menu Chức Năng")
choice = st.sidebar.radio(
    "Chọn:",
    ["1️⃣ Theo Tên Phim", "2️⃣ Theo AI", "3️⃣ Theo Thể Loại", "🚪 Đăng xuất"]
)

if choice == "🚪 Đăng xuất":
    st.session_state.user = None
    st.rerun()

# ================= MAIN =================
st.title(f"🎬 Chào mừng, {st.session_state.user}")

if choice == "1️⃣ Theo Tên Phim":
    movie = st.selectbox("Chọn phim", movies_df["Tên phim"])
    if st.button("Tìm đề xuất"):
        show_movies(content_based(movie))

elif choice == "2️⃣ Theo AI":
    st.subheader("🤖 AI gợi ý theo hồ sơ người dùng")
    base = movies_df.sample(1)["Tên phim"].values[0]
    show_movies(content_based(base))

elif choice == "3️⃣ Theo Thể Loại":
    if st.session_state.genres:
        show_movies(genre_based(st.session_state.genres))
    else:
        st.info("⚠️ Người dùng chưa có thể loại yêu thích")

