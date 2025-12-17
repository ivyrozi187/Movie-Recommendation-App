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
