import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ====================== FILE ======================
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"
GUEST_USER = "Guest"

# ====================== THEME (LIGHT – CHỮ ĐẬM) ======================
BG_COLOR = "#FAFAFA"
TEXT_COLOR = "#222222"
PRIMARY = "#FFAD7F"
ACCENT = "#C06C84"
CARD_BG = "#FFFFFF"

def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    h1,h2,h3 {{
        color: {ACCENT};
        font-weight: 800;
    }}

    /* ===== GENRE BUTTON ===== */
    .genre-btn {{
        border-radius: 14px;
        padding: 14px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #FFD6C9, #FFAD7F);
        color: #333;
        box-shadow: 0 6px 14px rgba(0,0,0,.15);
        transition: .3s;
    }}
    .genre-btn:hover {{
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 10px 22px rgba(0,0,0,.25);
        background: linear-gradient(135deg, #C06C84, #FFAD7F);
        color: white;
    }}

    /* ===== MOVIE CARD ===== */
    .movie-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill,minmax(250px,1fr));
        gap: 20px;
        margin-top: 20px;
    }}

    .movie-card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,.15);
        transition: .3s;
        border-left: 6px solid {PRIMARY};
    }}

    .movie-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 14px 30px rgba(0,0,0,.25);
    }}

    .movie-title {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {ACCENT};
    }}

    .movie-genre {{
        font-size: .85rem;
        color: #666;
        margin-bottom: 8px;
    }}

    .score {{
        font-weight: 800;
        color: {PRIMARY};
    }}
    </style>
    """, unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data
def load_data(path):
    return pd.read_csv(path).fillna("")

@st.cache_resource
def preprocess_movies():
    df = load_data(MOVIE_DATA_FILE)
    df["combined"] = df["Đạo diễn"] + " " + df["Diễn viên chính"] + " " + df["Thể loại phim"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    df["Độ phổ biến"] = pd.to_numeric(df["Độ phổ biến"], errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    df["pop_norm"] = scaler.fit_transform(df[["Độ phổ biến"]])

    df["parsed_genres"] = df["Thể loại phim"].apply(
        lambda x: set(g.strip() for g in x.split(","))
    )

    return df, cosine_sim

# ====================== SESSION ======================
if "user" not in st.session_state:
    st.session_state.user = None

if "used_movie_ids" not in st.session_state:
    st.session_state.used_movie_ids = set()

# ====================== UI HELPERS ======================
def show_cards(df, score_col):
    st.markdown("<div class='movie-grid'>", unsafe_allow_html=True)
    for _, r in df.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{r['Tên phim']}</div>
            <div class="movie-genre">{r['Thể loại phim']}</div>
            <div class="score">Điểm: {r[score_col]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ====================== AUTH ======================
def login_page(users):
    st.header("🔐 Đăng nhập")
    u = st.text_input("Tên người dùng")
    if st.button("Đăng nhập", type="primary"):
        if u in users["Tên người dùng"].values:
            st.session_state.user = u
            st.session_state.used_movie_ids = set()
            st.rerun()
        else:
            st.error("Tên người dùng không tồn tại")

    st.divider()
    if st.button("👀 Dùng thử"):
        st.session_state.user = GUEST_USER
        st.rerun()

# ====================== RECOMMEND ======================
def recommend_by_movie(movie, w):
    idx = movies[movies["Tên phim"] == movie].index[0]
    sims = list(enumerate(cosine_sim[idx]))
    df_sim = pd.DataFrame(sims, columns=["i", "sim"])
    df = movies.join(df_sim.set_index("i"))
    df["score"] = w * df["sim"] + (1 - w) * df["pop_norm"]
    return df.sort_values("score", ascending=False).iloc[1:11]

def recommend_by_ai(user):
    users = load_data(USER_DATA_FILE)
    row = users[users["Tên người dùng"] == user]
    genres = set(ast.literal_eval(row["5 phim coi gần nhất"].values[0]))

    df = movies.copy()
    df["score"] = df["parsed_genres"].apply(lambda x: len(x & genres))
    df = df[~df.index.isin(st.session_state.used_movie_ids)]
    result = df[df["score"] > 0].sort_values(
        ["score", "Độ phổ biến"], ascending=False
    ).head(10)

    st.session_state.used_movie_ids.update(result.index.tolist())
    return result

def zero_click():
    df = movies.copy()
    df["score"] = df["pop_norm"]
    return df.sort_values("score", ascending=False).head(10)

# ====================== MAIN ======================
inject_css()
movies, cosine_sim = preprocess_movies()
users = load_data(USER_DATA_FILE)

if st.session_state.user is None:
    login_page(users)
    st.stop()

st.sidebar.title("🎯 Chức năng")
menu = st.sidebar.radio(
    "Chọn:",
    ["Theo Tên Phim", "Theo AI", "Zero-Click", "Đăng xuất"]
)

st.title(f"🎬 Xin chào {st.session_state.user}")

if menu == "Theo Tên Phim":
    movie = st.selectbox("Chọn phim", movies["Tên phim"])
    w = st.slider("Trọng số Similarity", 0.0, 1.0, 0.7)
    recs = recommend_by_movie(movie, w)
    show_cards(recs, "score")

elif menu == "Theo AI":
    if st.button("🤖 Tìm Đề Xuất AI", type="primary"):
        recs = recommend_by_ai(st.session_state.user)
        show_cards(recs, "score")

elif menu == "Zero-Click":
    show_cards(zero_click(), "score")

elif menu == "Đăng xuất":
    st.session_state.user = None
    st.session_state.used_movie_ids = set()
   

