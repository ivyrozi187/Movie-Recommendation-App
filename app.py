import streamlit as st

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ================== FILE ==================
MOVIE_FILE = "data_phim_full_images.csv"
USER_FILE = "danh_sach_nguoi_dung_moi.csv"
GUEST = "Guest"

# ================== LOAD DATA ==================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path).fillna("")

@st.cache_resource
def preprocess_movies():
    df = load_csv(MOVIE_FILE)

    df["combined"] = (
        df["Đạo diễn"] + " " +
        df["Diễn viên chính"] + " " +
        df["Thể loại phim"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    df["Độ phổ biến"] = pd.to_numeric(df["Độ phổ biến"], errors="coerce").fillna(0)
    df["Năm phát hành"] = pd.to_numeric(df["Năm phát hành"], errors="coerce").fillna(2024)

    scaler = MinMaxScaler()
    df["popularity_norm"] = scaler.fit_transform(df[["Độ phổ biến"]])

    df["parsed_genres"] = df["Thể loại phim"].apply(
        lambda x: set(str(x).split(","))
    )

    return df, cosine_sim

# ================== SESSION ==================
if "user" not in st.session_state:
    st.session_state.user = None

if "users" not in st.session_state:
    st.session_state.users = load_csv(USER_FILE)

# ================== UI ==================
def inject_css():
    st.markdown("""
    <style>
        .movie-grid {
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
            gap:25px;
        }
        .card {
            background:white;
            border-radius:12px;
            box-shadow:0 6px 20px rgba(0,0,0,.15);
            overflow:hidden;
            transition:.3s;
        }
        .card:hover {
            transform:translateY(-8px);
            box-shadow:0 12px 30px rgba(0,188,212,.5);
        }
        .poster img {
            width:100%;
            height:300px;
            object-fit:cover;
        }
        .info { padding:15px }
        .score { color:#00BCD4; font-weight:800 }
    </style>
    """, unsafe_allow_html=True)

def show_movies(df, score_col):
    st.markdown('<div class="movie-grid">', unsafe_allow_html=True)
    for _, r in df.iterrows():
        st.markdown(f"""
        <div class="card">
            <div class="poster">
                <img src="{r['Poster']}">
            </div>
            <div class="info">
                <b>{r['Tên phim']}</b>
                <div class="score">{score_col}: {r[score_col]:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== AUTH ==================
def login():
    st.header("🔐 Đăng nhập")
    u = st.text_input("Tên người dùng")
    if st.button("Đăng nhập"):
        if u in st.session_state.users["Tên người dùng"].values:
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Không tồn tại")

def register():
    st.header("📝 Đăng ký")
    u = st.text_input("Tên mới")
    genres = st.multiselect(
        "Thể loại yêu thích",
        movies["Thể loại phim"].str.split(",").explode().unique()
    )
    if st.button("Đăng ký"):
        new = {
            "ID": len(st.session_state.users) + 1,
            "Tên người dùng": u,
            "5 phim coi gần nhất": str(genres),
            "Phim yêu thích nhất": ""
        }
        st.session_state.users = pd.concat(
            [st.session_state.users, pd.DataFrame([new])],
            ignore_index=True
        )
        st.session_state.user = u
        st.rerun()

# ================== RECOMMEND ==================
def recommend_by_movie(movie, w_sim):
    idx = movies[movies["Tên phim"] == movie].index[0]
    scores = list(enumerate(cosine_sim[idx]))

    df_sim = pd.DataFrame(scores, columns=["i", "sim"])
    df = movies.join(df_sim.set_index("i"))
    df["score"] = w_sim * df["sim"] + (1 - w_sim) * df["popularity_norm"]

    return df.sort_values("score", ascending=False).iloc[1:11]

def recommend_by_profile(user):
    row = st.session_state.users[
        st.session_state.users["Tên người dùng"] == user
    ]
    genres = set(ast.literal_eval(row["5 phim coi gần nhất"].values[0]))

    movies["score"] = movies["parsed_genres"].apply(
        lambda x: len(x.intersection(genres))
    )
    return movies[movies["score"] > 0].sort_values(
        ["score", "Độ phổ biến"], ascending=False
    ).head(10)

def zero_click():
    movies["score"] = (
        0.5 * movies["popularity_norm"]
        + 0.5 * (movies["Năm phát hành"] / movies["Năm phát hành"].max())
    )
    return movies.sort_values("score", ascending=False).head(10)

# ================== MAIN ==================
inject_css()
movies, cosine_sim = preprocess_movies()

if st.session_state.user is None:
    t1, t2, t3 = st.tabs(["Login", "Register", "Guest"])
    with t1: login()
    with t2: register()
    with t3:
        st.session_state.user = GUEST
        st.rerun()
    st.stop()

st.sidebar.title("🎯 Menu")
menu = st.sidebar.radio(
    "Chọn chức năng",
    ["Theo Tên Phim", "Theo AI", "Zero-Click", "Đăng xuất"]
)

st.title(f"🎬 Xin chào {st.session_state.user}")

if menu == "Theo Tên Phim":
    movie = st.selectbox("Chọn phim", movies["Tên phim"])
    w = st.slider("Trọng số Similarity", 0.0, 1.0, 0.7)
    recs = recommend_by_movie(movie, w)
    show_movies(recs, "score")

elif menu == "Theo AI":
    recs = recommend_by_profile(st.session_state.user)
    show_movies(recs, "score")

elif menu == "Zero-Click":
    recs = zero_click()
    show_movies(recs, "score")

elif menu == "Đăng xuất":
    st.session_state.user = None
    st.rerun()
