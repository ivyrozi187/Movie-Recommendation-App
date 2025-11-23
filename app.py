import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys

# --- CẤU HÌNH TÊN FILE ---
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- KHỞI TẠO BIẾN TRẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None

# --- KHỞI TẠO BIẾN TOÀN CỤC ---
df_movies = None
df_users = None
cosine_sim = None


# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU & HELPERS
# ==============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Tải và tiền xử lý dữ liệu cho cả hai hệ thống đề xuất."""
    try:
        # Tải dữ liệu phim
        df_movies = pd.read_csv(MOVIE_DATA_FILE).fillna("")
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # Tải dữ liệu người dùng
        df_users = pd.read_csv(USER_DATA_FILE).fillna("")
        df_users.columns = [col.strip() for col in df_users.columns]

        # Đảm bảo cột ID là số
        if 'ID' in df_users.columns:
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')

        # 1. Tiền xử lý cho Content-Based (TF-IDF/Cosine Sim)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến (để dùng cho hệ thống TF-IDF)
        scaler = MinMaxScaler()
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based (Genre Matching)
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        return df_movies, df_users, cosine_sim

    except FileNotFoundError as e:
        st.error(f"LỖI TẢI DỮ LIỆU: Không tìm thấy file {e.filename}. Vui lòng kiểm tra file.")
        st.stop()
    except KeyError as e:
        st.error(f"LỖI TÊN CỘT: Thiếu cột {e}. Vui lòng kiểm tra tên cột trong file CSV.")
        st.stop()
    except Exception as e:
        st.error(f"LỖI KHÔNG XÁC ĐỊNH trong quá trình tải/tiền xử lý dữ liệu: {e}")
        st.stop()


def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

def register_new_user_form(df_movies, df_users):
    """Form đăng ký người dùng mới."""
    st.header("📝 Đăng Ký Tài Khoản Mới")

    with st.form("register_form"):
        username = st.text_input("Tên người dùng mới (Duy nhất):").strip()

        st.subheader("Chọn Phim Đã Xem (Tối thiểu 5 phim để có hồ sơ tốt)")

        # Lấy danh sách phim cho selection box
        movie_titles_list = get_unique_movie_titles()

        # Chọn 5 phim gần nhất
        recent_list_raw = st.multiselect(
            "🎥 5 Phim Đã Xem Gần Nhất:",
            options=movie_titles_list,
            key='recent_select',
            default=movie_titles_list[:5]  # Mặc định 5 phim đầu tiên để dễ thử nghiệm
        )

        # Chọn phim yêu thích nhất
        favorite_movie = st.selectbox(
            "⭐ Phim Yêu Thích Nhất:",
            options=movie_titles_list,
            key='favorite_select'
        )

        submitted = st.form_submit_button("Đăng Ký & Đăng Nhập")

        if submitted:
            if not username:
                st.warning("Vui lòng nhập tên người dùng.")
                return

            if username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên người dùng '{username}' đã tồn tại.")
                return

            if len(recent_list_raw) < 5:
                st.warning("Vui lòng chọn tối thiểu 5 phim đã xem gần nhất.")
                return

            # 1. Tạo ID mới
            max_id = df_users['ID'].max()
            new_id = int(max_id) + 1 if pd.notna(max_id) else 1

            # 2. Lưu dữ liệu người dùng mới
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': [str(recent_list_raw)],  # Lưu dưới dạng chuỗi list
                'Phim yêu thích nhất': [favorite_movie]
            }
            new_user_df = pd.DataFrame(new_user_data)

            # Thêm vào DataFrame tổng và lưu lại
            df_users = pd.concat([df_users, new_user_df], ignore_index=True)
            df_users.to_csv(USER_DATA_FILE, index=False, encoding='utf-8')

            # 3. Đăng nhập và chuyển trạng thái
            st.session_state['logged_in_user'] = username
            st.success(f"🎉 Đăng ký và đăng nhập thành công! Chào mừng, {username}.")
            st.rerun()  # Tải lại ứng dụng để hiển thị Menu Chính


def login_form(df_users):
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")

    with st.form("login_form"):
        username = st.text_input("Tên người dùng:").strip()
        submitted = st.form_submit_button("Đăng Nhập")

        if submitted:
            if username in df_users['Tên người dùng'].values:
                st.session_state['logged_in_user'] = username
                st.success(f"✅ Đăng nhập thành công! Chào mừng, {username}.")
                st.rerun()
            else:
                st.error("❌ Tên người dùng không tồn tại.")


def authentication_page(df_movies, df_users):
    """Trang Xác thực (chọn Đăng nhập hoặc Đăng ký)."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Đăng Nhập"):
            st.session_state['auth_mode'] = 'login'
    with col2:
        if st.button("Đăng Ký"):
            st.session_state['auth_mode'] = 'register'

    if 'auth_mode' not in st.session_state or st.session_state['auth_mode'] == 'login':
        login_form(df_users)

    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies, df_users)


# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT (Logic)
# ==============================================================================

# (Các hàm đề xuất và vẽ biểu đồ được giữ nguyên logic)

def get_recommendations(username, df_users, df_movies, num_recommendations=10):
    """Đề xuất phim dựa trên 5 phim người dùng xem gần nhất và sở thích thể loại."""
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    try:
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = ast.literal_eval(watched_movies_str)
    except (ValueError, SyntaxError, IndexError):
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = [m.strip() for m in watched_movies_str.split(',') if m.strip()]

    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    watched_and_favorite = set(watched_list + [favorite_movie])

    watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
    user_genres = set()
    for genres in watched_genres['parsed_genres']:
        user_genres.update(genres)

    if not user_genres: return pd.DataFrame()

    candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()

    def calculate_score(candidate_genres):
        return len(candidate_genres.intersection(user_genres))

    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(calculate_score)

    recommended_df = candidate_movies.sort_values(
        by=['Similarity_Score', 'Độ phổ biến'],
        ascending=[False, False]
    )

    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)


def get_movie_index(movie_name):
    """Tìm chỉ mục của phim trong DataFrame."""
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
        return idx
    except IndexError:
        return -1


def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    """Đề xuất phim dựa trên sự kết hợp giữa độ giống (sim) và độ phổ biến (pop)."""
    idx = get_movie_index(movie_name)
    if idx == -1: return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])

    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    df_result['weighted_score'] = (
            weight_sim * df_result['similarity'] +
            weight_pop * df_result['popularity_norm']
    )

    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)

    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)


def plot_genre_popularity(movie_name, top_movies, df_users, df_movies, is_user_based=False):
    """Vẽ biểu đồ so sánh ĐỘ PHỔ BIẾN TRUNG BÌNH của các thể loại liên quan."""

    # 1. Thu thập dữ liệu thể loại và độ phổ biến
    if is_user_based:
        user_row = df_users[df_users['Tên người dùng'] == st.session_state['logged_in_user']]
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = ast.literal_eval(watched_movies_str)
        watched_df = df_movies[df_movies['Tên phim'].isin(watched_list)]

        combined_df = pd.concat([watched_df, top_movies], ignore_index=True)
        title = f"Độ Phổ Biến Thể Loại (Hồ sơ {st.session_state['logged_in_user']} & Đề xuất)"

    else:
        movie_row = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()]
        if movie_row.empty: return
        combined_df = pd.concat([movie_row, top_movies], ignore_index=True)
        title = f"Độ Phổ Biến TB của Các Thể Loại Phim Liên Quan đến '{movie_name}'"

    genres_data = []
    for index, row in combined_df.iterrows():
        for genre in row['Thể loại phim'].split(','):
            genres_data.append({
                'Thể loại': genre.strip(),
                'Độ phổ biến': row['Độ phổ biến']
            })

    df_plot = pd.DataFrame(genres_data)
    genre_avg_pop = df_plot.groupby('Thể loại')['Độ phổ biến'].mean().reset_index()
    top_7_genres = genre_avg_pop.sort_values(by='Độ phổ biến', ascending=False).head(7)
    overall_avg_pop = df_plot['Độ phổ biến'].mean()

    # 2. Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(top_7_genres['Thể loại'], top_7_genres['Độ phổ biến'],
                  color='skyblue', edgecolor='black', alpha=0.8)

    ax.axhline(overall_avg_pop, color='red', linestyle='--', linewidth=1.5,
               label=f'TB Tổng thể ({overall_avg_pop:.1f})')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, round(yval, 1), ha='center', fontsize=10, weight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Thể loại phim")
    ax.set_ylabel("Độ Phổ Biến Trung Bình (Popularity Score)")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)


def get_unique_movie_titles():
    """Lấy danh sách các tên phim duy nhất."""
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, df_users, cosine_sim):
    st.title(f"🎬 Chào mừng, {st.session_state['logged_in_user']}!")

    st.sidebar.title("Menu Đề Xuất")

    # Tạo lựa chọn menu trên sidebar
    menu_choice = st.sidebar.radio(
        "Chọn chức năng:",
        ('Đề xuất theo Tên Phim', 'Đề xuất theo Hồ Sơ', 'Đăng Xuất')
    )

    if st.sidebar.button("Đăng Xuất", key="logout_btn"):
        st.session_state['logged_in_user'] = None
        st.experimental_rerun()

    st.sidebar.write("-" * 20)

    if menu_choice == 'Đề xuất theo Tên Phim':
        st.header("1️⃣ Đề xuất dựa trên Nội dung (TF-IDF)")

        movie_titles_list = get_unique_movie_titles()
        movie_name = st.selectbox("🎥 Chọn tên phim bạn yêu thích:", options=movie_titles_list)

        weight_sim = st.slider("⚖️ Trọng số Độ giống (Similarity)", 0.0, 1.0, 0.7, 0.1)
        weight_pop = 1 - weight_sim

        if st.button("Tìm Đề Xuất", key="find_sim"):
            result = recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim)

            if not result.empty:
                st.subheader(f"🎬 10 Đề xuất phim dựa trên '{movie_name}':")
                st.dataframe(result, use_container_width=True)

                if st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại"):
                    plot_genre_popularity(movie_name,
                                          df_movies[df_movies['Tên phim'].isin(result['Tên phim'].tolist())],
                                          df_users, df_movies, is_user_based=False)
            else:
                st.warning("⚠️ Không tìm thấy đề xuất hoặc phim gốc không tồn tại.")

    elif menu_choice == 'Đề xuất theo Hồ Sơ':
        st.header("2️⃣ Đề xuất dựa trên Hồ sơ Người dùng")

        username = st.session_state['logged_in_user']
        user_row = df_users[df_users['Tên người dùng'] == username]

        # Hiển thị 5 phim đã xem gần nhất
        recent_films_str = user_row['5 phim coi gần nhất'].iloc[0]
        recent_films = recent_films_str.strip('[]').replace("'", "")
        st.info(f"5 Phim đã xem gần nhất: {recent_films}")

        if st.button("Tìm Đề Xuất Hồ Sơ", key="find_profile"):
            recommendations = get_recommendations(username, df_users, df_movies, num_recommendations=10)

            if not recommendations.empty:
                st.subheader(f"✅ 10 Đề xuất Phim Dành Cho Bạn:")
                st.dataframe(recommendations, use_container_width=True)

                if st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", key="plot_profile_check"):
                    plot_genre_popularity(None,
                                          df_movies[df_movies['Tên phim'].isin(recommendations['Tên phim'].tolist())],
                                          df_users, df_movies, is_user_based=True)
            else:
                st.warning("⚠️ Không có đề xuất nào được tạo. Kiểm tra dữ liệu thể loại phim đã xem.")


# ==============================================================================
# V. CHẠY ỨNG DỤNG CHÍNH
# ==============================================================================

if __name__ == '__main__':
    # Tải và xử lý dữ liệu một lần
    df_movies, df_users, cosine_sim = load_and_preprocess_data()

    # Định tuyến trang
    if st.session_state['logged_in_user']:
        main_page(df_movies, df_users, cosine_sim)
    else:
        authentication_page(df_movies, df_users)