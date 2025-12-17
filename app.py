import streamlit as st

# ================== BẮT BUỘC: PAGE CONFIG Ở ĐẦU ==================
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

# ================== FILE ==================
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "data_phim_full_images.csv"
GUEST_USER = "Guest"

# ================== LOAD DATA AN TOÀN ==================
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path).fillna("")
    except:
        st.error(f"❌ Không tìm thấy file: {path}")
        return pd.DataFrame()

# ================== THEME ==================
def inject_light_theme():
    st.markdown("""
    <style>
        .stApp { background-color:#F7F9FC; }
        h1,h2,h3 { font-weight:800; }
        .movie-grid {
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
            gap:25px;
            margin-top:20px;
        }
        .movie-card {
            background:white;
            border-radius:12px;
            box-shadow:0 6px 20px rgba(0,0,0,.15);
            transition:.3s;
            overflow:hidden;
        }
        .movie-card:hover {
            transform:translateY(-8px);
            box-shadow:0 12px 30px rgba(0,188,212,.5);
        }
        .poster {
            height:300px;
            background:#E0F7FA;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:40px;
        }
        .info {
            padding:15px;
        }
        .score {
            color:#00BCD4;
            font-weight:800;
        }
    </style>
    """, unsafe_allow_html=True)

# ================== HELPERS ==================
def parse_genres(s):
    return set(g.strip() for g in s.split(",") if g.strip())

# ================== PREPROCESS ==================
@st.cache_resource
def preprocess_movies():
    df = load_data(MOVIE_DATA_FILE)
    if df.empty:
        return df, np.array([[]])

    df["combined"] = df["Đạo diễn"] + " " + df["Diễn viên chính"] + " " + df["Thể loại phim"]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(matrix)

    df["Độ phổ biến"] = pd.to_numeric(df["Độ phổ biến"], errors="coerce").fillna(0)
    df["parsed_genres"] = df["Thể loại phim"].apply(parse_genres)

    return df, cosine_sim

# ================== SESSION ==================
if "user" not in st.session_state:
    st.session_state.user = None
if "users" not in st.session_state:
    st.session_state.users = load_data(USER_DATA_FILE)

# ================== AUTH ==================
def login_page():
    st.header("🔐 Đăng nhập")

    username = st.text_input("Tên người dùng")
    if st.button("Đăng nhập", type="primary"):
        if username in st.session_state.users["Tên người dùng"].values:
            st.session_state.user = username
            st.rerun()
        else:
            st.error("❌ Không tồn tại")

    st.divider()
    if st.button("👀 Dùng thử không cần đăng nhập"):
        st.session_state.user = GUEST_USER
        st.rerun()

def register_page():
    st.header("📝 Đăng ký")

    username = st.text_input("Tên người dùng mới")
    genres = st.multiselect(
        "Chọn thể loại yêu thích",
        movies["Thể loại phim"].str.split(",").explode().unique()
    )

    if st.button("Đăng ký", type="primary"):
        if not username or not genres:
            st.error("Thiếu thông tin")
            return

        new = {
            "ID": len(st.session_state.users) + 1,
            "Tên người dùng": username,
            "5 phim coi gần nhất": str(genres),
            "Phim yêu thích nhất": ""
        }
        st.session_state.users = pd.concat(
            [st.session_state.users, pd.DataFrame([new])],
            ignore_index=True
        )
        st.session_state.user = username
        st.success("🎉 Thành công")
        st.rerun()

# ================== DISPLAY GRID ==================
def show_movies(df, score_col):
    st.markdown('<div class="movie-grid">', unsafe_allow_html=True)
    for _, r in df.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="poster">🎬</div>
            <div class="info">
                <b>{r['Tên phim']}</b><br>
                <span class="score">{score_col}: {r[score_col]:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== MAIN ==================
inject_light_theme()
movies, cosine_sim = preprocess_movies()

if st.session_state.user is None:
    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
    with tab1: login_page()
    with tab2: register_page()
    st.stop()

# ================== APP ==================
st.title(f"🎬 Chào mừng {st.session_state.user}")

movie_list = movies["Tên phim"].tolist()
selected = st.selectbox("Chọn phim bạn thích", movie_list)

idx = movies[movies["Tên phim"] == selected].index[0]
scores = list(enumerate(cosine_sim[idx]))
df_sim = pd.DataFrame(scores, columns=["i", "score"])
df_rec = movies.join(df_sim.set_index("i"), how="left").sort_values("score", ascending=False)[1:11]

show_movies(df_rec, "score")

if st.button("🚪 Đăng xuất"):
    st.session_state.user = None
    st.rerun()
