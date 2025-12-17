import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# CONFIG
# ======================================================
st.set_page_config("MovieFlix", layout="wide")

MOVIE_FILE = "data_phim_full_images.csv"
USER_FILE = "user_dataset_ready.csv"   # đổi nếu bạn dùng file khác

for f in [MOVIE_FILE, USER_FILE]:
    if not os.path.exists(f):
        st.error(f"❌ Thiếu file: {f}")
        st.stop()

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_movies():
    df = pd.read_csv(MOVIE_FILE).fillna("")
    df["combined"] = (
        df["Đạo diễn"].astype(str) + " " +
        df["Diễn viên chính"].astype(str) + " " +
        df["Thể loại phim"].astype(str)
    )
    df["Độ phổ biến"] = pd.to_numeric(df["Độ phổ biến"], errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    df["pop_norm"] = scaler.fit_transform(df[["Độ phổ biến"]])
    return df

@st.cache_data
def load_users():
    return pd.read_csv(USER_FILE)

movies_df = load_movies()
users_df = load_users()

@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(df["combined"])
    return cosine_similarity(mat)

cosine_sim = build_similarity(movies_df)

# ======================================================
# SESSION
# ======================================================
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ======================================================
# CSS – UI ĐẸP HƠN
# ======================================================
st.markdown("""
<style>
body { background:#141414; color:white; }
h1,h2,h3,p { color:white; }
.movie-card {
    transition: transform .3s;
    cursor:pointer;
}
.movie-card:hover {
    transform: scale(1.12);
}
.poster {
    border-radius:14px;
}
.hero {
    height:360px;
    background-size:cover;
    border-radius:16px;
    margin-bottom:20px;
    display:flex;
    align-items:flex-end;
    padding:30px;
    font-size:42px;
    font-weight:bold;
    text-shadow:2px 2px 8px black;
}
.carousel {
    display:flex;
    overflow-x:auto;
    gap:18px;
    padding:10px 0;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HELPER UI
# ======================================================
def movie_card(movie):
    with st.container():
        if st.button(" ", key=movie["Tên phim"]):
            st.session_state.selected_movie = movie["Tên phim"]
            st.session_state.page = "detail"
            st.rerun()
        st.image(movie["Link Backdrop"], use_container_width=True)
        st.caption(movie["Tên phim"])

def carousel(df, title):
    st.markdown(f"## {title}")
    cols = st.columns(min(6, len(df)))
    for col, (_, row) in zip(cols, df.iterrows()):
        with col:
            if st.button(" ", key=f"{row['Tên phim']}_{title}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.session_state.page = "detail"
                st.rerun()
            st.image(row["Link Backdrop"], use_container_width=True)
            st.caption(row["Tên phim"])

# ======================================================
# LOGIN
# ======================================================
if not st.session_state.user:
    st.title("🎬 MovieFlix")
    username = st.selectbox("Chọn người dùng", users_df["username"])
    if st.button("Đăng nhập"):
        st.session_state.user = username
        st.rerun()
    st.stop()

user = users_df[users_df["username"] == st.session_state.user].iloc[0]

# ======================================================
# MOVIE DETAIL PAGE
# ======================================================
if st.session_state.page == "detail":
    movie = movies_df[movies_df["Tên phim"] == st.session_state.selected_movie].iloc[0]

    st.image(movie["Link Backdrop"], use_container_width=True)
    st.title(movie["Tên phim"])

    col1, col2 = st.columns([2,3])
    with col1:
        st.write("🎭 **Thể loại:**", movie["Thể loại phim"])
        st.write("🎬 **Đạo diễn:**", movie["Đạo diễn"])
        st.write("⭐ **Diễn viên:**", movie["Diễn viên chính"])
        st.write("📅 **Năm:**", movie["Năm phát hành"])
        st.write("🔥 **Độ phổ biến:**", movie["Độ phổ biến"])

    with col2:
        st.subheader("🎯 Phim tương tự")
        idx = movie.name
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_df = movies_df.copy()
        sim_df["sim"] = [s for _, s in sim_scores]
        sim_df = sim_df.sort_values("sim", ascending=False).iloc[1:7]
        carousel(sim_df, "Phim tương tự")

    if st.button("⬅️ Quay lại"):
        st.session_state.page = "home"
        st.rerun()

    st.stop()

# ======================================================
# HOME PAGE
# ======================================================
st.markdown(
    f"""
    <div class="hero" style="background-image:url('{movies_df.iloc[0]['Link Backdrop']}')">
        Chào mừng {user['username']}
    </div>
    """,
    unsafe_allow_html=True
)

# Phim đã xem
recent_titles = user["recent_movies"].split("|")
recent_df = movies_df[movies_df["Tên phim"].isin(recent_titles)]
carousel(recent_df, "🎞️ Phim đã xem gần đây")

# Gợi ý theo hồ sơ
fav = user["favorite_movie"]
genre = movies_df[movies_df["Tên phim"] == fav]["Thể loại phim"].iloc[0].split(",")[0]
rec_df = movies_df[movies_df["Thể loại phim"].str.contains(genre, na=False)].head(6)
carousel(rec_df, "⭐ Gợi ý cho bạn")
