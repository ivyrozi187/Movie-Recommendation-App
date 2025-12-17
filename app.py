import streamlit as st

# ================== PAGE CONFIG (BẮT BUỘC Ở ĐẦU) ==================
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
import numpy as np

# ================== FILE ==================
MOVIE_DATA_FILE = "data_phim_full_images.csv"

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv(MOVIE_DATA_FILE).fillna("")
    return df

# ================== THEME + CSS ==================
def inject_css():
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
        .movie-poster img {
            width:100%;
            height:300px;
            object-fit:cover;
        }
        .movie-info {
            padding:15px;
        }
        .score {
            color:#00BCD4;
            font-weight:800;
        }
        .genre {
            font-size:13px;
            color:#666;
        }
    </style>
    """, unsafe_allow_html=True)

# ================== HELPERS ==================
def get_all_genres(df):
    genres = set()
    for g in df["Thể loại phim"]:
        for item in str(g).split(","):
            genres.add(item.strip())
    return sorted(list(genres))

def display_movies(df, score_col):
    st.markdown('<div class="movie-grid">', unsafe_allow_html=True)

    for _, row in df.iterrows():
        title = row["Tên phim"]
        genres = row["Thể loại phim"]
        score = row[score_col]
        poster = row.get("Poster", "")

        poster_html = (
            f"<img src='{poster}'>" if poster else "<div style='height:300px;display:flex;align-items:center;justify-content:center;'>🎬</div>"
        )

        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-poster">
                {poster_html}
            </div>
            <div class="movie-info">
                <b>{title}</b>
                <div class="genre">{genres}</div>
                <div class="score">Score: {score:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================== MAIN APP ==================
inject_css()
movies = load_data()

st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM THEO THỂ LOẠI")
st.caption("Chọn nhiều thể loại bạn yêu thích, hệ thống sẽ đề xuất phim phù hợp nhất")

# --------- CHỌN THỂ LOẠI ---------
all_genres = get_all_genres(movies)

selected_genres = st.multiselect(
    "🎭 Chọn thể loại (có thể chọn nhiều):",
    options=all_genres,
    placeholder="Ví dụ: Phim Hành Động, Phim Chiến Tranh, Phim Khoa Học Viễn Tưởng"
)

# --------- ĐỀ XUẤT ---------
if selected_genres:
    st.info(f"🔍 Đề xuất cho thể loại: {', '.join(selected_genres)}")

    def genre_score(movie_genres):
        movie_genres = set(str(movie_genres).split(","))
        return len(movie_genres.intersection(selected_genres))

    movies["genre_score"] = movies["Thể loại phim"].apply(genre_score)

    recommendations = (
        movies[movies["genre_score"] > 0]
        .sort_values(by=["genre_score", "Độ phổ biến"], ascending=False)
        .head(12)
    )

    if recommendations.empty:
        st.warning("❌ Không tìm thấy phim phù hợp")
    else:
        display_movies(recommendations, "genre_score")

else:
    st.warning("👉 Vui lòng chọn ít nhất 1 thể loại")
