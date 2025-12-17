import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
import random

# ===================== CONFIG =====================
st.set_page_config("MovieFlix", layout="wide")

# ===================== FILE CHECK =====================
MOVIE_FILE = "movie_info_1000.csv"
USER_FILE = "user_dataset_ready.csv"

for f in [MOVIE_FILE, USER_FILE]:
    if not os.path.exists(f):
        st.error(f"❌ Thiếu file: {f}")
        st.stop()

# ===================== LOAD DATA =====================
@st.cache_data
def load_movies():
    df = pd.read_csv(MOVIE_FILE).fillna("")
    df["combined"] = df["Đạo diễn"] + " " + df["Diễn viên chính"] + " " + df["Thể loại phim"]
    df["Độ phổ biến"] = pd.to_numeric(df["Độ phổ biến"], errors="coerce").fillna(df["Độ phổ biến"].mean())
    scaler = MinMaxScaler()
    df["pop_norm"] = scaler.fit_transform(df[["Độ phổ biến"]])
    return df

@st.cache_data
def load_users():
    return pd.read_csv(USER_FILE)

movies_df = load_movies()
users_df = load_users()

# ===================== TF-IDF =====================
@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    return cosine_similarity(matrix)

cosine_sim = build_similarity(movies_df)

# ===================== SESSION =====================
if "user" not in st.session_state:
    st.session_state.user = None
if "guest_topics" not in st.session_state:
    st.session_state.guest_topics = []

# ===================== CSS (NETFLIX STYLE) =====================
st.markdown("""
<style>
body { background-color: #141414; }
h1,h2,h3,p { color:white; }
[data-testid="stImage"] {
    border-radius: 12px;
    transition: transform .2s;
}
[data-testid="stImage"]:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

# ===================== LOGIN / GUEST =====================
def login_page():
    st.title("🎬 MovieFlix")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if username in users_df["username"].values:
                st.session_state.user = username
                st.rerun()
            else:
                st.error("User không tồn tại")

    with col2:
        if st.button("🚀 Dùng thử (Guest)"):
            st.session_state.user = "GUEST"
            st.rerun()

# ===================== ZERO CLICK =====================
TOPICS = {
    "Marvel": ["Action", "Sci-Fi"],
    "Sitcom": ["Comedy"],
    "Cổ Trang": ["History", "Drama"],
    "Xuyên Không": ["Fantasy", "Sci-Fi"]
}

def zero_click_recommend(genres):
    pattern = "|".join(genres)
    return movies_df[movies_df["Thể loại phim"].str.contains(pattern, na=False)] \
        .sort_values("Độ phổ biến", ascending=False).head(10)

# ===================== CONTENT BASED =====================
def content_based(movie_name, w_sim):
    idx = movies_df[movies_df["Tên phim"] == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    df = pd.DataFrame(scores, columns=["idx","sim"])
    df = df.merge(movies_df, left_on="idx", right_index=True)
    df["score"] = w_sim * df["sim"] + (1 - w_sim) * df["pop_norm"]
    return df.sort_values("score", ascending=False).head(10)

# ===================== PROFILE BASED =====================
def profile_based(user):
    fav = users_df[users_df["username"] == user]["favorite_movie"].iloc[0]
    genres = movies_df[movies_df["Tên phim"] == fav]["Thể loại phim"].iloc[0]
    return movies_df[movies_df["Thể loại phim"].str.contains(genres.split(",")[0], na=False)] \
        .sort_values("Độ phổ biến", ascending=False).head(10)

# ===================== MAIN =====================
if not st.session_state.user:
    login_page()
    st.stop()

st.sidebar.title("🎬 Menu")

if st.sidebar.button("Đăng xuất"):
    st.session_state.user = None
    st.rerun()

# ===================== GUEST MODE =====================
if st.session_state.user == "GUEST":
    st.header("🔥 Bạn quan tâm gì?")
    cols = st.columns(len(TOPICS))

    for col, t in zip(cols, TOPICS):
        with col:
            if st.button(t):
                st.session_state.guest_topics = TOPICS[t]

    if st.session_state.guest_topics:
        rec = zero_click_recommend(st.session_state.guest_topics)
        st.subheader("🎯 Gợi ý cho bạn")
        st.dataframe(rec[["Tên phim","Thể loại phim","Độ phổ biến"]])

        fig, ax = plt.subplots()
        rec.groupby("Thể loại phim")["Độ phổ biến"].mean().head(5).plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ===================== USER MODE =====================
else:
    user = users_df[users_df["username"] == st.session_state.user].iloc[0]

    st.title(f"🍿 Xin chào {user['username']}")

    movies = user["recent_movies"].split("|")
    imgs = user["recent_images"].split("|")

    st.subheader("🎞️ Phim đã xem")
    cols = st.columns(5)
    for c,m,i in zip(cols,movies,imgs):
        c.image(i, use_container_width=True)
        c.caption(m)

    st.subheader("❤️ Phim yêu thích")
    st.image(user["favorite_image"], width=260)
    st.write(user["favorite_movie"])

    tab1, tab2 = st.tabs(["🎥 Theo phim", "👤 Theo hồ sơ"])

    with tab1:
        movie = st.selectbox("Chọn phim", movies_df["Tên phim"])
        w = st.slider("Similarity", 0.0, 1.0, 0.7)
        if st.button("Gợi ý"):
            rec = content_based(movie, w)
            st.dataframe(rec[["Tên phim","score","Độ phổ biến"]])

    with tab2:
        if st.button("Gợi ý theo hồ sơ"):
            rec = profile_based(st.session_state.user)
            st.dataframe(rec[["Tên phim","Thể loại phim","Độ phổ biến"]])
