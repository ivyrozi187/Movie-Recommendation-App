# ===============================
# DreamStream - Movie Recommendation System
# FULL VERSION: Rating + Content-based + Profile-based
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="DreamStream - Movie Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# FILE PATHS
# ===============================
MOVIE_DATA_FILE = "movie_info_1000.csv"
IMAGE_DATA_FILE = "data_phim_full_images.csv"
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
RATING_FILE = "ratings.csv"
GUEST_USER = "Guest"

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_movies():
    df = pd.read_csv(MOVIE_DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    df['movie_id'] = df.index
    return df.fillna("")

@st.cache_data
def load_images():
    try:
        return pd.read_csv(IMAGE_DATA_FILE)
    except:
        return pd.DataFrame(columns=["Tên phim", "image_url"])

@st.cache_data
def load_users():
    if os.path.exists(USER_DATA_FILE):
        df = pd.read_csv(USER_DATA_FILE)
    else:
        df = pd.DataFrame(columns=["ID", "Tên người dùng", "5 phim coi gần nhất", "Phim yêu thích nhất"])
    return df

@st.cache_data
def load_ratings():
    if os.path.exists(RATING_FILE):
        return pd.read_csv(RATING_FILE)
    return pd.DataFrame(columns=["user_id", "movie_id", "rating"])

# ===============================
# PREPROCESS
# ===============================
def parse_genres(s):
    return set([g.strip() for g in s.split(',') if g.strip()])

@st.cache_resource
def preprocess_movies(df):
    df = df.copy()
    df['combined'] = df['Đạo diễn'] + " " + df['Diễn viên chính'] + " " + df['Thể loại phim']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    df['Độ phổ biến'] = pd.to_numeric(df['Độ phổ biến'], errors='coerce').fillna(0)
    scaler = MinMaxScaler()
    df['pop_norm'] = scaler.fit_transform(df[['Độ phổ biến']])

    df['parsed_genres'] = df['Thể loại phim'].apply(parse_genres)
    return df, cosine_sim

# ===============================
# IMAGE + RATING HELPERS
# ===============================
def get_image(movie_name):
    row = df_images[df_images['Tên phim'] == movie_name]
    if not row.empty:
        return row['image_url'].iloc[0]
    return None


def save_rating(user_id, movie_id, rating):
    df = load_ratings()
    mask = (df.user_id == user_id) & (df.movie_id == movie_id)
    if mask.any():
        df.loc[mask, 'rating'] = rating
    else:
        df = pd.concat([
            df,
            pd.DataFrame([[user_id, movie_id, rating]], columns=df.columns)
        ])
    df.to_csv(RATING_FILE, index=False)
    st.cache_data.clear()


def get_avg_rating(movie_id):
    df = load_ratings()
    r = df[df.movie_id == movie_id]
    return None if r.empty else round(r.rating.mean(), 2)

# ===============================
# RECOMMENDATION FUNCTIONS
# ===============================
def recommend_by_movie(title, df, cosine_sim, top=10):
    idx = df[df['Tên phim'] == title].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top+1]
    ids = [i[0] for i in scores]
    df_res = df.iloc[ids].copy()
    df_res['score'] = [i[1] for i in scores]
    return df_res


def recommend_by_profile(username, df):
    user = df_users[df_users['Tên người dùng'] == username]
    if user.empty:
        return pd.DataFrame()

    genres = []
    try:
        genres = ast.literal_eval(user['5 phim coi gần nhất'].iloc[0])
    except:
        pass

    df['match'] = df['parsed_genres'].apply(lambda g: len(g.intersection(genres)))
    return df.sort_values(['match', 'Độ phổ biến'], ascending=False).head(10)

# ===============================
# UI COMPONENTS
# ===============================
def movie_card(row, username=None):
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            img = get_image(row['Tên phim'])
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown("🎬")
        with col2:
            st.subheader(row['Tên phim'])
            st.caption(row['Thể loại phim'])
            avg = get_avg_rating(row.movie_id)
            if avg:
                st.write(f"⭐ {avg}/5")

            if username and username != GUEST_USER:
                rating = st.slider(
                    "Đánh giá",
                    1.0, 5.0, 3.0, 0.5,
                    key=f"rate_{row.movie_id}"
                )
                if st.button("Lưu", key=f"save_{row.movie_id}"):
                    uid = df_users[df_users['Tên người dùng'] == username]['ID'].iloc[0]
                    save_rating(uid, row.movie_id, rating)
                    st.success("Đã lưu đánh giá")

# ===============================
# MAIN APP
# ===============================
df_movies = load_movies()
df_images = load_images()
df_users = load_users()
df_movies, cosine_sim = preprocess_movies(df_movies)

if 'user' not in st.session_state:
    st.session_state.user = None

st.sidebar.title("DreamStream")

if not st.session_state.user:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập") and u in df_users['Tên người dùng'].values:
            st.session_state.user = u
            st.rerun()

    with tab2:
        u = st.text_input("Tên mới")
        g = st.text_input("Thể loại yêu thích (vd: ['Phim Hành Động'])")
        if st.button("Đăng ký"):
            new = pd.DataFrame([[len(df_users)+1, u, g, ""]], columns=df_users.columns)
            df_users = pd.concat([df_users, new])
            df_users.to_csv(USER_DATA_FILE, index=False)
            st.session_state.user = u
            st.rerun()

    with tab3:
        if st.button("Vào với tư cách khách"):
            st.session_state.user = GUEST_USER
            st.rerun()

else:
    st.title(f"Xin chào {st.session_state.user}")
    menu = st.sidebar.radio("Chức năng", ["Theo phim", "Theo hồ sơ", "Đăng xuất"])

    if menu == "Theo phim":
        movie = st.selectbox("Chọn phim", df_movies['Tên phim'])
        if st.button("Đề xuất"):
            res = recommend_by_movie(movie, df_movies, cosine_sim)
            for _, r in res.iterrows():
                movie_card(r, st.session_state.user)

    elif menu == "Theo hồ sơ":
        res = recommend_by_profile(st.session_state.user, df_movies)
        for _, r in res.iterrows():
            movie_card(r, st.session_state.user)

    else:
        st.session_state.user = None
        st.rerun()
