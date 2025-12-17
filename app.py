# ===============================
# DreamStream - Movie Recommendation System (FULL, FIXED)
# Profile-based = dựa trên LỊCH SỬ PHIM ĐÃ XEM GẦN NHẤT
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ===============================
# PAGE CONFIG (CHỈ GỌI 1 LẦN)
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
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
IMAGE_DATA_FILE = "data_phim_full_images.csv"
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
def load_users():
    if USER_DATA_FILE:
        df = pd.read_csv(USER_DATA_FILE)
    else:
        df = pd.DataFrame(columns=["ID","Tên người dùng","5 phim coi gần nhất","Phim yêu thích nhất"])
    return df

@st.cache_data
def load_images():
    try:
        return pd.read_csv(IMAGE_DATA_FILE)
    except:
        return pd.DataFrame()

# ===============================
# PREPROCESS
# ===============================
def parse_genres(s):
    if not isinstance(s,str): return set()
    return set([g.strip() for g in s.split(',') if g.strip()])

@st.cache_resource
def preprocess(df):
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
# IMAGE HELPER (AUTO-DETECT COLUMN)
# ===============================
def get_image(movie_name):
    if df_images.empty: return None
    row = df_images[df_images['Tên phim'] == movie_name]
    if row.empty: return None
    for col in df_images.columns:
        if col.lower() in ['image','image_url','poster','poster_url','img','link_anh']:
            val = row[col].iloc[0]
            if isinstance(val,str) and val.startswith('http'):
                return val
    return None

# ===============================
# RECOMMENDATION LOGIC
# ===============================
def recommend_by_movie(title, df, cosine_sim, top=10):
    idx = df[df['Tên phim'] == title].index
    if len(idx)==0: return pd.DataFrame()
    idx = idx[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x:x[1], reverse=True)[1:top+1]
    ids = [i[0] for i in scores]
    res = df.iloc[ids].copy()
    res['score'] = [i[1] for i in scores]
    return res

# ===== PROFILE-BASED (DỰA TRÊN LỊCH SỬ PHIM ĐÃ XEM) =====
def build_user_profile(username):
    row = df_users[df_users['Tên người dùng']==username]
    if row.empty: return set(), set()

    try:
        watched = set(ast.literal_eval(row['5 phim coi gần nhất'].iloc[0]))
    except:
        watched = set()

    genres = set()
    for m in watched:
        g = df_movies[df_movies['Tên phim']==m]['parsed_genres']
        if not g.empty:
            genres |= g.iloc[0]

    return watched, genres


def recommend_by_profile(username, df, top=10):
    watched, genres = build_user_profile(username)

    if not genres:
        return df.sort_values('Độ phổ biến', ascending=False).head(top)

    df = df[~df['Tên phim'].isin(watched)].copy()
    df['profile_score'] = df['parsed_genres'].apply(lambda g: len(g & genres))

    return df.sort_values(['profile_score','Độ phổ biến'], ascending=False).head(top)

# ===============================
# UI COMPONENT
# ===============================
def movie_card(row, username):
    col1, col2 = st.columns([1,2])
    with col1:
        img = get_image(row['Tên phim'])
        if img:
            st.image(img, use_container_width=True)
        else:
            st.markdown("🎬")
    with col2:
        st.subheader(row['Tên phim'])
        st.caption(row['Thể loại phim'])

# ===============================
# MAIN APP
# ===============================
df_movies = load_movies()
df_users = load_users()
df_images = load_images()
df_movies, cosine_sim = preprocess(df_movies)

if 'user' not in st.session_state:
    st.session_state.user = None

st.sidebar.title("DreamStream")

# ===== AUTH =====
if not st.session_state.user:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập","Đăng ký","Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập") and u in df_users['Tên người dùng'].values:
            st.session_state.user = u
            st.rerun()

    with tab2:
        u = st.text_input("Tên mới")
        watched = st.text_input("5 phim coi gần nhất (vd: ['Avatar','Titanic'])")
        if st.button("Đăng ký"):
            new = pd.DataFrame([[len(df_users)+1,u,watched,""]], columns=df_users.columns)
            df_users = pd.concat([df_users,new])
            df_users.to_csv(USER_DATA_FILE, index=False)
            st.session_state.user = u
            st.rerun()

    with tab3:
        if st.button("Vào với tư cách khách"):
            st.session_state.user = GUEST_USER
            st.rerun()

# ===== MAIN =====
else:
    st.title(f"Xin chào {st.session_state.user}")
    menu = st.sidebar.radio("Chức năng", ["Theo phim","Theo hồ sơ","Đăng xuất"])

    if menu=="Theo phim":
        movie = st.selectbox("Chọn phim", df_movies['Tên phim'])
        if st.button("Đề xuất"):
            res = recommend_by_movie(movie, df_movies, cosine_sim)
            for _,r in res.iterrows():
                movie_card(r, st.session_state.user)

    elif menu=="Theo hồ sơ":
        if st.session_state.user==GUEST_USER:
            st.warning("Khách không có hồ sơ")
        else:
            res = recommend_by_profile(st.session_state.user, df_movies)
            for _,r in res.iterrows():
                movie_card(r, st.session_state.user)

    else:
        st.session_state.user=None
        st.rerun()
