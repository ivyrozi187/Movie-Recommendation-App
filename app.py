import streamlit as st
import pandas as pd

st.set_page_config(page_title="MovieFlix", layout="wide")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv("user_dataset_ready.csv")
    return df

users_df = load_data()

# ========== NAVBAR ==========
st.markdown("""
<style>
.navbar {
    background-color: #141414;
    padding: 15px;
}
.navbar h1 {
    color: #E50914;
    margin: 0;
    font-size: 30px;
}
</style>
<div class="navbar">
    <h1>MovieFlix</h1>
</div>
""", unsafe_allow_html=True)

# ========== SELECT USER ==========
user_id = st.selectbox("Chọn người dùng", users_df["user_id"])
user = users_df[users_df["user_id"] == user_id].iloc[0]

# ========== HERO BANNER ==========
st.markdown("""
<style>
.hero {
    background-image: url('https://images.unsplash.com/photo-1606761560503-b7a8e4f0f4d6');
    background-size: cover;
    background-position: center;
    height: 350px;
    border-radius: 8px;
}
.hero h2 {
    color: white;
    padding: 150px 30px;
    font-size: 48px;
    text-shadow: 2px 2px 6px black;
}
</style>
<div class="hero">
    <h2>Đề xuất cho bạn</h2>
</div>
""", unsafe_allow_html=True)

# ========== MOVIE ROW ==========
def movie_row(title, movie_list, image_list):
    st.markdown(f"### {title}")
    cols = st.columns(len(movie_list))
    for col, m, img in zip(cols, movie_list, image_list):
        with col:
            st.image(img, use_container_width=True)
            st.write(m)

# ========== RECENTLY WATCHED ==========
recent_movies = user["recent_movies"].split("|")
recent_images = user["recent_images"].split("|")
movie_row("Phim đã xem gần nhất", recent_movies, recent_images)

# ========== FAVORITE ==========
st.markdown("### Phim yêu thích")
st.image(user["favorite_image"], width=240)
st.write(user["favorite_movie"])

# ========== SIMPLE RECOMMENDATIONS ==========
other = users_df[users_df["user_id"] != user_id].sample(5).iloc[0]
rec_movies = other["recent_movies"].split("|")
rec_images = other["recent_images"].split("|")
movie_row("Gợi ý cho bạn", rec_movies, rec_images)

# ========== END ==========
st.markdown("---")
st.write("🚀 Ứng dụng gợi ý phim tích hợp phong cách Netflix")
