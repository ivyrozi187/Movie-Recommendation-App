import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# CONFIG
# =========================================================
st.set_page_config("MovieFlix", layout="wide")

MOVIE_FILE = "movie_info_1000.csv"
USER_FILE = "user_dataset_ready.csv"

for f in [MOVIE_FILE, USER_FILE]:
    if not os.path.exists(f):
        st.error(f"❌ Thiếu file: {f}")
        st.stop()

# =========================================================
# THEME STATE
# =========================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = None
if "theme_locked" not in st.session_state:
    st.session_state.theme_locked = False

def apply_theme(dark=True):
    if dark:
        st.markdown("""
        <style>
        body { background:#141414; color:white; transition:0.4s; }
        h1,h2,h3,p,span { color:white; }
        .carousel{display:flex;overflow-x:auto;gap:16px;padding:10px}
        .card{min-width:180px;transition:.3s;cursor:pointer}
        .card:hover{transform:scale(1.15)}
        .card img{width:180px;border-radius:12px}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body { background:#f7f7f7; color:#111; transition:0.4s; }
        h1,h2,h3,p,span { color:#111; }
        .carousel{display:flex;overflow-x:auto;gap:16px;padding:10px}
        .card{min-width:180px;transition:.3s;cursor:pointer}
        .card:hover{transform:scale(1.15)}
        .card img{width:180px;border-radius:12px}
        </style>
        """, unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
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

@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    return cosine_similarity(matrix)

cosine_sim = build_similarity(movies_df)

# =========================================================
# SESSION
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None
if "ai_rec_history" not in st.session_state:
    st.session_state.ai_rec_history = set()
if "ai_offset" not in st.session_state:
    st.session_state.ai_offset = 0

# =========================================================
# UI HELPERS
# =========================================================
def carousel(title, movies, images):
    st.markdown(f"### {title}")
    html = '<div class="carousel">'
    for m,i in zip(movies, images):
        html += f"""
        <div class="card">
            <img src="{i}">
            <div style="text-align:center;font-size:.85rem">{m}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# LOGIN / GUEST
# =========================================================
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

# =========================================================
# ZERO CLICK
# =========================================================
TOPICS = {
    "Marvel": ["Action","Sci-Fi"],
    "Sitcom": ["Comedy"],
    "Cổ Trang": ["History","Drama"],
    "Xuyên Không": ["Fantasy","Sci-Fi"]
}

def zero_click(genres):
    return movies_df[movies_df["Thể loại phim"].str.contains("|".join(genres), na=False)] \
        .sort_values("Độ phổ biến", ascending=False).head(10)

# =========================================================
# CONTENT BASED
# =========================================================
def content_based(movie, w):
    idx = movies_df[movies_df["Tên phim"]==movie].index[0]
    scores = cosine_sim[idx]
    df = movies_df.copy()
    df["score"] = w*scores + (1-w)*df["pop_norm"]
    return df.sort_values("score", ascending=False).head(10)

# =========================================================
# PROFILE BASED
# =========================================================
def profile_based(user):
    fav = users_df[users_df["username"]==user]["favorite_movie"].iloc[0]
    genre = movies_df[movies_df["Tên phim"]==fav]["Thể loại phim"].iloc[0].split(",")[0]
    return movies_df[movies_df["Thể loại phim"].str.contains(genre, na=False)] \
        .sort_values("Độ phổ biến", ascending=False).head(10)

# =========================================================
# AI SEARCH
# =========================================================
def ai_search(user_row):
    watched = user_row["recent_movies"].split("|")
    idxs = movies_df[movies_df["Tên phim"].isin(watched)].index.tolist()
    if not idxs: return pd.DataFrame()

    sim = cosine_sim[idxs].mean(axis=0)
    df = movies_df.copy()
    df["sim"] = sim
    df = df[~df["Tên phim"].isin(watched)]
    df = df[~df["Tên phim"].isin(st.session_state.ai_rec_history)]
    df = df.sort_values(["sim","Độ phổ biến"], ascending=False)

    batch = df.iloc[st.session_state.ai_offset:st.session_state.ai_offset+10]
    st.session_state.ai_offset += 10
    st.session_state.ai_rec_history.update(batch["Tên phim"].tolist())
    return batch

# =========================================================
# MAIN
# =========================================================
st.sidebar.title("🎨 Giao diện")
st.session_state.dark_mode = st.sidebar.toggle(
    "🌙 Dark mode",
    value=True if st.session_state.dark_mode in [None,True] else False
)

apply_theme(st.session_state.dark_mode!=False)

if not st.session_state.user:
    login_page()
    st.stop()

if st.sidebar.button("Đăng xuất"):
    st.session_state.user=None
    st.session_state.ai_rec_history=set()
    st.session_state.ai_offset=0
    st.rerun()

# ================= GUEST =================
if st.session_state.user=="GUEST":
    st.header("🔥 Bạn đang quan tâm gì?")
    cols=st.columns(len(TOPICS))
    for c,t in zip(cols,TOPICS):
        with c:
            if st.button(t):
                rec=zero_click(TOPICS[t])
                st.dataframe(rec[["Tên phim","Thể loại phim","Độ phổ biến"]])
    st.stop()

# ================= USER =================
user = users_df[users_df["username"]==st.session_state.user].iloc[0]

st.title(f"🍿 Xin chào {user['username']}")

carousel(
    "🎞️ Phim đã xem",
    user["recent_movies"].split("|"),
    user["recent_images"].split("|")
)

st.subheader("❤️ Phim yêu thích")
st.image(user["favorite_image"], width=250)
st.write(user["favorite_movie"])

tab1,tab2,tab3 = st.tabs(["🎥 Theo phim","👤 Theo hồ sơ","🔮 AI Search"])

with tab1:
    m=st.selectbox("Chọn phim",movies_df["Tên phim"])
    w=st.slider("Similarity",0.0,1.0,0.7)
    if st.button("Gợi ý"):
        r=content_based(m,w)
        st.dataframe(r[["Tên phim","Độ phổ biến","score"]])

with tab2:
    if st.button("Gợi ý theo hồ sơ"):
        r=profile_based(st.session_state.user)
        st.dataframe(r[["Tên phim","Thể loại phim","Độ phổ biến"]])

with tab3:
    if st.button("🤖 AI tìm phim"):
        st.session_state.ai_rec_history=set()
        st.session_state.ai_offset=0
        r=ai_search(user)
        st.session_state.ai_result=r
    if st.button("🔄 Tìm thêm 10 phim"):
        r=ai_search(user)
        st.session_state.ai_result=r
    if "ai_result" in st.session_state:
        rec=st.session_state.ai_result
        carousel(
            "🎯 AI đề xuất",
            rec["Tên phim"].tolist(),
            ["https://via.placeholder.com/300x450"]*len(rec)
        )
