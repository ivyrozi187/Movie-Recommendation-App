import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DreamStream", layout="wide")

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
# PREPROCESS
# ======================================================
def safe_col(df, col):
    return df[col].astype(str) if col in df.columns else ""

movies_df["content"] = (
    safe_col(movies_df, "Thể loại phim") + " " +
    safe_col(movies_df, "Diễn viên") + " " +
    safe_col(movies_df, "Đạo diễn")
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["content"])
cosine_sim = cosine_similarity(tfidf_matrix)

# ======================================================
# SESSION STATE (ĐÚNG KIỂU)
# ======================================================
defaults = {
    "logged_in_user": None,
    "selected_movie": None,
    "last_results": None,
    "guest_genres": [],
    "is_new_user": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# UTILS
# ======================================================
def get_all_genres():
    s = set()
    for g in movies_df["Thể loại phim"]:
        for x in str(g).split(","):
            s.add(x.strip())
    return sorted(s)

ALL_GENRES = get_all_genres()

def get_poster(row):
    if "Link Poster" in row and str(row["Link Poster"]).startswith("http"):
        return row["Link Poster"]
    return "https://via.placeholder.com/300x450?text=No+Image"

def show_movies(df):
    if df is None or df.empty:
        st.info("Không có phim để hiển thị")
        return
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(get_poster(row), use_container_width=True)
            st.caption(row["Tên phim"])
            if st.button("🎬 Xem chi tiết", key=f"{row['Tên phim']}_{i}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.rerun()

# ======================================================
# RECOMMEND LOGIC
# ======================================================
def content_based(title, top_n=10):
    if title not in movies_df["Tên phim"].values:
        return movies_df.sample(top_n)
    idx = movies_df[movies_df["Tên phim"] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies_df.iloc[[i[0] for i in scores]]

def recommend_by_genres(genres, top_n=10):
    if not genres:
        return movies_df.sample(top_n)
    df = movies_df[movies_df["Thể loại phim"].apply(
        lambda x: any(g in x for g in genres)
    )]
    return df.sample(min(len(df), top_n)) if not df.empty else movies_df.sample(top_n)

def profile_based(user_row, top_n=10):
    try:
        watched = ast.literal_eval(user_row["5 phim coi gần nhất"])
    except:
        watched = []

    watched = watched[:5]
    if not watched:
        return movies_df.sample(top_n)

    genres = movies_df[movies_df["Tên phim"].isin(watched)]["Thể loại phim"]
    if genres.empty:
        return movies_df.sample(top_n)

    main = genres.str.split(",").explode().value_counts().idxmax()
    return movies_df[movies_df["Thể loại phim"].str.contains(main, na=False)].sample(top_n)

# ======================================================
# LOGIN
# ======================================================
if st.session_state.logged_in_user is None:
    st.title("🍿 DreamStream")
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập") and u in users_df["Tên người dùng"].values:
            st.session_state.logged_in_user = u
            st.rerun()

    with tab2:
        new = st.text_input("Tên người dùng mới")
        g = st.multiselect("Thể loại yêu thích", ALL_GENRES)
        if st.button("Hoàn tất") and new:
            st.session_state.logged_in_user = new
            st.session_state.guest_genres = g
            st.session_state.is_new_user = True
            st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại xem", ALL_GENRES)
        if st.button("Vào chế độ Khách"):
            st.session_state.logged_in_user = "GUEST"
            st.session_state.guest_genres = g
            st.rerun()

    st.stop()

# ======================================================
# DETAIL
# ======================================================
if st.session_state.selected_movie:
    m = movies_df[movies_df["Tên phim"] == st.session_state.selected_movie].iloc[0]
    st.image(get_poster(m), use_container_width=True)
    st.title(m["Tên phim"])
    st.write("🎭", m["Thể loại phim"])
    show_movies(content_based(m["Tên phim"], 5))
    if st.button("⬅️ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()
    st.stop()

# ======================================================
# MENU
# ======================================================
menu = st.sidebar.radio(
    "Menu",
    ["Cá nhân", "Đề xuất theo Tên Phim", "Đề xuất theo AI", "Đề xuất theo Thể loại", "Đăng xuất"]
)

if menu == "Đăng xuất":
    st.session_state.clear()
    st.rerun()

st.session_state.last_results = None

# ======================================================
# SCREENS
# ======================================================
if menu == "Cá nhân":
    if st.session_state.logged_in_user == "GUEST":
        st.info("Khách không có trang cá nhân")
    else:
        user = users_df[users_df["Tên người dùng"] == st.session_state.logged_in_user].iloc[0]
        try:
            recent = ast.literal_eval(user["5 phim coi gần nhất"])[:5]
        except:
            recent = []
        show_movies(movies_df[movies_df["Tên phim"].isin(recent)])

elif menu == "Đề xuất theo Tên Phim":
    movie = st.selectbox("Chọn phim", movies_df["Tên phim"])
    if st.button("Đề xuất"):
        st.session_state.last_results = content_based(movie)

elif menu == "Đề xuất theo AI":
    if st.button("Đề xuất AI"):
        if st.session_state.logged_in_user == "GUEST":
            st.session_state.last_results = recommend_by_genres(st.session_state.guest_genres)
        else:
            user = users_df[users_df["Tên người dùng"] == st.session_state.logged_in_user].iloc[0]
            st.session_state.last_results = profile_based(user)

elif menu == "Đề xuất theo Thể loại":
    st.session_state.last_results = recommend_by_genres(st.session_state.guest_genres)

# ======================================================
# SHOW RESULT
# ======================================================
if st.session_state.last_results is not None:
    st.markdown("---")
    show_movies(st.session_state.last_results)
