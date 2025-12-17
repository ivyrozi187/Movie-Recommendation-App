import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ==============================================================================
# 1. CẤU HÌNH TRANG
# ==============================================================================
st.set_page_config(
    page_title="Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD & XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")

    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')

    movies['combined_features'] = (
        movies['Tên phim'] + " " +
        movies['Đạo diễn'] + " " +
        movies['Thể loại phim']
    )

    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 🔒 CHỈ LẤY 5 PHIM GẦN NHẤT
    users['history_list'] = users['5 phim coi gần nhất'].apply(
        lambda x: ast.literal_eval(x)[:5] if isinstance(x, str) else []
    )

    all_genres = set()
    for g in movies['Thể loại phim']:
        for x in g.split(','):
            all_genres.add(x.strip())

    return movies, users, cosine_sim, sorted(list(all_genres))

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. BIỂU ĐỒ THỐNG KÊ CÁ NHÂN (CHỈ 5 PHIM ĐÃ XEM)
# ==============================================================================
def draw_user_charts(history_titles):
    """
    Biểu đồ xu hướng xem phim cá nhân
    DỮ LIỆU = 5 PHIM ĐÃ XEM GẦN NHẤT
    """

    history_titles = history_titles[:5]

    if not history_titles:
        st.warning("Người dùng chưa có lịch sử xem phim.")
        return

    genres = []
    for title in history_titles:
        row = movies_df[movies_df['Tên phim'] == title]
        if not row.empty:
            genres.extend(
                [g.strip() for g in row.iloc[0]['Thể loại phim'].split(',')]
            )

    if not genres:
        st.warning("Không đủ dữ liệu thể loại để vẽ biểu đồ.")
        return

    counter = Counter(genres)
    df_chart = (
        pd.DataFrame(counter.items(), columns=['Thể loại', 'Số phim'])
        .sort_values(by='Số phim', ascending=False)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    ax1.pie(
        df_chart['Số phim'],
        labels=df_chart['Thể loại'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title("Tỷ lệ thể loại đã xem (5 phim gần nhất)")

    # Bar chart
    sns.barplot(
        x='Số phim',
        y='Thể loại',
        data=df_chart,
        ax=ax2
    )
    ax2.set_title("Xu hướng xem phim (5 phim gần nhất)")

    st.pyplot(fig)

# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎬 DreamStream")

    if st.session_state.user_mode == 'member':
        menu = st.radio(
            "Chức năng",
            ["Đề xuất AI", "Tìm kiếm Phim", "Theo Thể loại Yêu thích", "Thống kê Cá nhân"]
        )
        if st.button("Đăng xuất"):
            st.session_state.clear()
            st.rerun()
    else:
        menu = "Login"

# ==============================================================================
# 6. LOGIN
# ==============================================================================
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên người dùng")
        if st.button("Đăng nhập"):
            row = users_df[users_df['Tên người dùng'] == u]
            if not row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = row.iloc[0]
                st.rerun()

    with tab2:
        new_user = st.text_input("Tên mới")
        g = st.multiselect("Chọn thể loại", ALL_GENRES)
        if st.button("Đăng ký"):
            st.session_state.user_mode = 'register'
            st.session_state.user_genres = g
            st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại xem", ALL_GENRES)
        if st.button("Vào khách"):
            st.session_state.user_mode = 'guest'
            st.session_state.user_genres = g
            st.rerun()

# ==============================================================================
# 7. MEMBER – THỐNG KÊ CÁ NHÂN
# ==============================================================================
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user.get('history_list', [])

    if menu == "Thống kê Cá nhân":
        st.header("📊 Xu hướng Xem phim Cá nhân")
        draw_user_charts(user_history)
