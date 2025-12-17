import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
st.set_page_config(page_title="DreamStream", layout="wide")

# ================= LOAD DATA =================
@st.cache_data
def load_movies():
    return pd.read_csv("data_phim_full_images.csv").fillna("")

@st.cache_data
def load_users():
    return pd.read_csv("danh_sach_nguoi_dung_moi.csv").fillna("")

movies = load_movies()
users = load_users()

# ================= SAFE COLUMN =================
def safe_col(df, col):
    if col in df.columns:
        return df[col].astype(str)
    return ""

# ================= CONTENT =================
movies["content"] = (
    safe_col(movies, "Thể loại phim") + " " +
    safe_col(movies, "Diễn viên chính") + " " +
    safe_col(movies, "Diễn viên") + " " +
    safe_col(movies, "Đạo diễn")
)

# ================= TF-IDF =================
tfidf = TfidfVectorizer(stop_words="english")
cosine_sim = cosine_similarity(tfidf.fit_transform(movies["content"]))

# ================= SESSION =================
for k in ["user", "selected_movie", "results", "user_genres", "guest_genres"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ================= HELPERS =================
def poster(row):
    for c in ["Link Poster", "Link Backdrop"]:
        if c in row and str(row[c]).startswith("http"):
            return row[c]
    return "https://via.placeholder.com/300x450?text=No+Image"

def show_movies(df):
    cols = st.columns(5)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(poster(row), use_container_width=True)
            st.caption(row["Tên phim"])
            if st.button("🎬 Xem chi tiết", key=f"d_{i}_{row['Tên phim']}"):
                st.session_state.selected_movie = row["Tên phim"]
                st.rerun()

def recommend_by_genres(genres, n=10):
    df = movies[movies["Thể loại phim"].apply(lambda x: any(g in x for g in genres))]
    return df.sample(min(n, len(df))) if not df.empty else movies.sample(n)

def plot_trend(movie_list):
    genres = []
    for m in movie_list:
        row = movies[movies["Tên phim"] == m]
        if not row.empty:
            genres += row.iloc[0]["Thể loại phim"].split(",")

    if not genres:
        st.info("Không đủ dữ liệu vẽ biểu đồ")
        return

    c = Counter([g.strip() for g in genres])
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(c.keys(), c.values(), color="#f4a7b9")
    ax.set_title("Xu hướng xem phim")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ================= LOGIN / REGISTER / GUEST =================
if st.session_state.user is None:
    st.title("🍿 DreamStream")
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Chế độ Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            if u in users["Tên người dùng"].values:
                st.session_state.user = u
                st.rerun()
            else:
                st.error("❌ Không tồn tại")

    with tab2:
        new_user = st.text_input("Tên người dùng mới")
        genres = st.multiselect(
            "Chọn thể loại yêu thích",
            sorted({g.strip() for x in movies["Thể loại phim"] for g in x.split(",")})
        )
        if st.button("Hoàn tất & Đề xuất"):
            if new_user and len(genres) >= 2:
                st.session_state.user = new_user
                st.session_state.user_genres = genres
                st.session_state.results = recommend_by_genres(genres)
                st.rerun()

    with tab3:
        g = st.multiselect(
            "Chọn thể loại muốn xem",
            sorted({g.strip() for x in movies["Thể loại phim"] for g in x.split(",")})
        )
        if st.button("Vào với tư cách Khách"):
            if g:
                st.session_state.user = "GUEST"
                st.session_state.guest_genres = g
                st.session_state.results = recommend_by_genres(g)
                st.rerun()

    st.stop()

# ================= DETAIL =================
if st.session_state.selected_movie:
    m = movies[movies["Tên phim"] == st.session_state.selected_movie].iloc[0]
    st.image(poster(m), use_container_width=True)
    st.title(m["Tên phim"])
    st.write("🎭", m["Thể loại phim"])
    if st.button("⬅ Quay lại"):
        st.session_state.selected_movie = None
        st.rerun()
    st.stop()

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "Menu",
    ["Cá nhân", "Tìm phim", "Theo thể loại", "Đăng xuất"]
)

if menu == "Đăng xuất":
    st.session_state.clear()
    st.rerun()

st.header(f"🎬 Xin chào {st.session_state.user}")

# ================= CÁ NHÂN =================
if menu == "Cá nhân" and st.session_state.user != "GUEST":
    user = users[users["Tên người dùng"] == st.session_state.user].iloc[0]
    try:
        recent = ast.literal_eval(user["5 phim coi gần nhất"])[:5]
    except:
        recent = []

    rows = [movies[movies["Tên phim"] == m].iloc[0]
            for m in recent if not movies[movies["Tên phim"] == m].empty]

    show_movies(pd.DataFrame(rows))
    plot_trend(recent)

# ================= SEARCH =================
elif menu == "Tìm phim":
    q = st.text_input("Nhập chính xác tên phim")
    if q:
        r = movies[movies["Tên phim"].str.lower() == q.lower()]
        if not r.empty:
            show_movies(r)
        else:
            st.warning("Không tìm thấy")

# ================= GENRE =================
elif menu == "Theo thể loại":
    genres = st.session_state.guest_genres if st.session_state.user == "GUEST" else st.session_state.user_genres
    if genres:
        st.session_state.results = recommend_by_genres(genres)
        if st.button("🔄 Tạo đề xuất mới"):
            st.session_state.results = recommend_by_genres(genres)

# ================= SHOW =================
if st.session_state.results is not None and menu != "Cá nhân":
    st.markdown("---")
    show_movies(st.session_state.results)
