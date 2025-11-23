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
# Lưu ý: Cần có 2 file CSV này trong cùng thư mục:
# 1. danh_sach_nguoi_dung_moi.csv (chứa ID, Tên người dùng, 5 phim coi gần nhất, Phim yêu thích nhất)
# 2. movie_info_1000.csv (chứa thông tin phim: Tên phim, Đạo diễn, Diễn viên chính, Thể loại phim, Độ phổ biến,...)
USER_DATA_FILE = "danh_sach_nguoi_dung_moi.csv"
MOVIE_DATA_FILE = "movie_info_1000.csv"

# --- KHỞI TẠO BIẾN TRẠNG THÁI (SESSION STATE) ---
if 'logged_in_user' not in st.session_state:
    st.session_state['logged_in_user'] = None
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'login'
# Khởi tạo các biến để lưu kết quả và trạng thái hiển thị biểu đồ
if 'last_sim_result' not in st.session_state: st.session_state['last_sim_result'] = pd.DataFrame()
if 'last_sim_movie' not in st.session_state: st.session_state['last_sim_movie'] = None
if 'show_sim_plot' not in st.session_state: st.session_state['show_sim_plot'] = False

if 'last_profile_recommendations' not in st.session_state: st.session_state['last_profile_recommendations'] = pd.DataFrame()
if 'show_profile_plot' not in st.session_state: st.session_state['show_profile_plot'] = False


# ==============================================================================
# I. PHẦN TIỀN XỬ LÝ DỮ LIỆU & HELPERS
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """Hàm helper để tải dữ liệu CSV với cache."""
    return pd.read_csv(file_path).fillna("")

def parse_genres(genre_string):
    """Chuyển chuỗi thể loại thành tập hợp genres."""
    if not isinstance(genre_string, str) or not genre_string:
        return set()
    genres = [g.strip().replace('"', '') for g in genre_string.split(',')]
    return set(genres)
    
@st.cache_resource # Chỉ tải dữ liệu tĩnh một lần
def load_and_preprocess_static_data():
    """Tải và tiền xử lý dữ liệu tĩnh (movies và mô hình)."""
    try:
        df_movies = load_data(MOVIE_DATA_FILE)
        df_movies.columns = [col.strip() for col in df_movies.columns]

        # 1. Tiền xử lý cho Content-Based (TF-IDF/Cosine Sim)
        df_movies["combined_features"] = (
                df_movies["Đạo diễn"] + " " +
                df_movies["Diễn viên chính"] + " " +
                df_movies["Thể loại phim"]
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_movies["combined_features"])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Chuẩn hóa Độ phổ biến
        # Đảm bảo cột 'Độ phổ biến' là số, xử lý lỗi nếu cần (coerce)
        df_movies['Độ phổ biến'] = pd.to_numeric(df_movies['Độ phổ biến'], errors='coerce')
        # Thay thế các giá trị NaN bằng giá trị trung bình trước khi scale
        mean_popularity = df_movies['Độ phổ biến'].mean() if not df_movies['Độ phổ biến'].empty else 0
        df_movies['Độ phổ biến'] = df_movies['Độ phổ biến'].fillna(mean_popularity)
        
        scaler = MinMaxScaler()
        # Reshape the column for MinMaxScaler
        df_movies["popularity_norm"] = scaler.fit_transform(df_movies[["Độ phổ biến"]])

        # 2. Tiền xử lý cho User-Based
        df_movies['parsed_genres'] = df_movies['Thể loại phim'].apply(parse_genres)

        return df_movies, cosine_sim_matrix

    except Exception as e:
        st.error(f"LỖI TẢI HOẶC XỬ LÝ DỮ LIỆU TĨNH: {e}. Vui lòng kiểm tra các file CSV.")
        # Dừng ứng dụng nếu có lỗi nghiêm trọng
        st.stop()


def initialize_user_data():
    """Khởi tạo hoặc tải dữ liệu người dùng vào Session State."""
    if 'df_users' not in st.session_state:
        try:
            df_users = load_data(USER_DATA_FILE)
            df_users.columns = [col.strip() for col in df_users.columns]
            df_users['ID'] = pd.to_numeric(df_users['ID'], errors='coerce')
            
            # Xử lý các giá trị NaN/NaT trong cột 'ID'
            df_users = df_users.dropna(subset=['ID'])
        except Exception:
            # Tạo DataFrame trống nếu file không tồn tại hoặc bị lỗi
            df_users = pd.DataFrame(columns=['ID', 'Tên người dùng', '5 phim coi gần nhất', 'Phim yêu thích nhất'])

        st.session_state['df_users'] = df_users
    
    return st.session_state['df_users']

def get_unique_movie_titles(df_movies):
    """Lấy danh sách các tên phim duy nhất."""
    return df_movies['Tên phim'].dropna().unique().tolist()


# ==============================================================================
# II. CHỨC NĂNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

# --- CALLBACK FUNCTIONS ---
def set_auth_mode(mode):
    """Hàm callback để chuyển đổi giữa Đăng nhập và Đăng ký."""
    st.session_state['auth_mode'] = mode
    # Reset các kết quả đề xuất khi chuyển trang
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    # XÓA BỎ st.rerun() VÌ STREAMLIT TỰ ĐỘNG CHẠY LẠI
    # st.rerun() 

def logout():
    """Hàm callback để Đăng xuất."""
    st.session_state['logged_in_user'] = None
    st.session_state['auth_mode'] = 'login'
    # Reset các kết quả đề xuất khi đăng xuất
    st.session_state['last_sim_result'] = pd.DataFrame()
    st.session_state['last_profile_recommendations'] = pd.DataFrame()
    # XÓA BỎ st.rerun() VÌ STREAMLIT TỰ ĐỘNG CHẠY LẠI
    # st.rerun() 
# ---------------------------

def register_new_user_form(df_movies):
    """Form đăng ký người dùng mới (Lưu vào bộ nhớ Streamlit)."""
    st.header("📝 Đăng Ký Tài Khoản Mới (Phiên Tạm Thời)")
    st.info("📢 Người dùng mới sẽ chỉ tồn tại trong phiên làm việc hiện tại của bạn.")

    df_users = st.session_state['df_users']
    movie_titles_list = get_unique_movie_titles(df_movies)

    with st.form("register_form"):
        username = st.text_input("Tên người dùng mới (Duy nhất):").strip()

        st.subheader("Chọn Phim Đã Xem (Tối thiểu 5 phim để có hồ sơ tốt)")
        
        recent_list_raw = st.multiselect(
            "🎥 5 Phim Đã Xem Gần Nhất:",
            options=movie_titles_list,
            key='recent_select',
            default=movie_titles_list[:5] if len(movie_titles_list) >= 5 else []
        )
        
        favorite_movie = st.selectbox(
            "⭐ Phim Yêu Thích Nhất:",
            options=movie_titles_list,
            key='favorite_select'
        )
        
        submitted = st.form_submit_button("Đăng Ký & Đăng Nhập")

        if submitted:
            # 1. Kiểm tra đầu vào
            if not username:
                st.error("Vui lòng nhập tên người dùng.")
                return
            
            if username in df_users['Tên người dùng'].values:
                st.error(f"❌ Tên người dùng '{username}' đã tồn tại.")
                return
            
            if len(recent_list_raw) < 5:
                st.warning("Vui lòng chọn tối thiểu 5 phim đã xem gần nhất.")
                return
            
            # 2. Tạo ID mới
            max_id = df_users['ID'].max() if not df_users.empty and pd.notna(df_users['ID'].max()) else 0
            new_id = int(max_id) + 1
            
            # 3. Tạo dữ liệu mới
            new_user_data = {
                'ID': [new_id],
                'Tên người dùng': [username],
                '5 phim coi gần nhất': [str(recent_list_raw)], # Lưu dưới dạng chuỗi list để dễ đọc/tải lại
                'Phim yêu thích nhất': [favorite_movie]
            }
            new_user_df = pd.DataFrame(new_user_data)
            
            # 4. CẬP NHẬT SESSION STATE (KHÔNG GHI FILE)
            st.session_state['df_users'] = pd.concat([df_users, new_user_df], ignore_index=True)
            
            # 5. Đăng nhập
            st.session_state['logged_in_user'] = username
            st.success(f"🎉 Đăng ký và đăng nhập thành công! Chào mừng, {username}.")
            st.rerun() 

def login_form():
    """Form đăng nhập."""
    st.header("🔑 Đăng Nhập")
    
    df_users = st.session_state['df_users']

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

def authentication_page(df_movies):
    """Trang Xác thực (chọn Đăng nhập hoặc Đăng ký)."""
    st.title("🎬 HỆ THỐNG ĐỀ XUẤT PHIM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SỬ DỤNG on_click, Streamlit sẽ tự động chạy lại
        st.button("Đăng Nhập", key="btn_login", on_click=set_auth_mode, args=('login',))
    with col2:
        # SỬ DỤNG on_click, Streamlit sẽ tự động chạy lại
        st.button("Đăng Ký", key="btn_register", on_click=set_auth_mode, args=('register',))

    if st.session_state['auth_mode'] == 'login':
        login_form()
    
    elif st.session_state['auth_mode'] == 'register':
        register_new_user_form(df_movies)

# ==============================================================================
# III. CHỨC NĂNG ĐỀ XUẤT & VẼ BIỂU ĐỒ
# ==============================================================================

def get_recommendations(username, df_movies, num_recommendations=10):
    """Đề xuất phim dựa trên 5 phim người dùng xem gần nhất và sở thích thể loại."""
    df_users = st.session_state['df_users']
    user_row = df_users[df_users['Tên người dùng'] == username]
    if user_row.empty: return pd.DataFrame()

    # Lấy chuỗi phim đã xem
    watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
    watched_list = []
    
    # 1. Cố gắng phân tích cú pháp chuỗi Python list (dành cho người dùng mới đăng ký)
    try:
        watched_list = ast.literal_eval(watched_movies_str)
        if not isinstance(watched_list, list):
             watched_list = [] # Đặt lại nếu không phải list
    except (ValueError, SyntaxError):
        # 2. Nếu thất bại, cố gắng phân tách bằng dấu phẩy và làm sạch (dành cho người dùng từ file CSV cũ)
        watched_list = [m.strip().strip("'") for m in watched_movies_str.strip('[]').split(',') if m.strip()]
    
    # Đảm bảo tất cả phần tử đều là chuỗi
    watched_list = [str(item) for item in watched_list if str(item).strip()]


    favorite_movie = user_row['Phim yêu thích nhất'].iloc[0]
    watched_and_favorite = set(watched_list + [favorite_movie])

    # Xác định các thể loại yêu thích của người dùng
    # Đảm bảo rằng watched_list chỉ chứa các tên phim hợp lệ
    watched_genres = df_movies[df_movies['Tên phim'].isin(watched_list)]
    user_genres = set()
    for genres in watched_genres['parsed_genres']:
        user_genres.update(genres)

    if not user_genres: 
        return pd.DataFrame()

    # Loại trừ các phim đã xem/yêu thích
    candidate_movies = df_movies[~df_movies['Tên phim'].isin(watched_and_favorite)].copy()

    # Tính điểm dựa trên số lượng thể loại chung
    def calculate_score(candidate_genres):
        return len(candidate_genres.intersection(user_genres))

    candidate_movies['Similarity_Score'] = candidate_movies['parsed_genres'].apply(calculate_score)

    # Kết hợp điểm tương đồng thể loại và độ phổ biến
    recommended_df = candidate_movies.sort_values(
        by=['Similarity_Score', 'Độ phổ biến'],
        ascending=[False, False]
    )

    return recommended_df[['Tên phim', 'Thể loại phim', 'Độ phổ biến', 'Similarity_Score']].head(num_recommendations)

def get_movie_index(movie_name, df_movies):
    """Tìm chỉ mục của phim trong DataFrame."""
    try:
        idx = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()].index[0]
        return idx
    except IndexError:
        return -1

def recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim):
    """Đề xuất phim dựa trên sự kết hợp giữa độ giống (sim) và độ phổ biến (pop)."""
    idx = get_movie_index(movie_name, df_movies)
    if idx == -1: return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'similarity'])

    df_result = pd.merge(df_movies, sim_scores_df, left_index=True, right_on='index')

    # Tính điểm trọng số kết hợp
    df_result['weighted_score'] = (
            weight_sim * df_result['similarity'] +
            weight_pop * df_result['popularity_norm']
    )

    # Loại bỏ phim gốc
    df_result = df_result.drop(df_result[df_result['Tên phim'] == movie_name].index)
    df_result = df_result.sort_values(by='weighted_score', ascending=False)

    return df_result[['Tên phim', 'weighted_score', 'similarity', 'Độ phổ biến', 'Thể loại phim']].head(10)

def plot_genre_popularity(movie_name, recommended_movies_df, df_movies, is_user_based=False):
    """Vẽ biểu đồ so sánh ĐỘ PHỔ BIẾN TRUNG BÌNH của các thể loại liên quan."""

    # 1. Thu thập dữ liệu thể loại và độ phổ biến
    df_users = st.session_state['df_users']
    
    combined_df = recommended_movies_df.copy() # Bắt đầu với các phim được đề xuất
    
    if is_user_based:
        user_row = df_users[df_users['Tên người dùng'] == st.session_state['logged_in_user']]
        # Xử lý chuỗi list để lấy danh sách phim đã xem (Sử dụng logic robust từ get_recommendations)
        watched_movies_str = user_row['5 phim coi gần nhất'].iloc[0]
        watched_list = []
        try:
            watched_list = ast.literal_eval(watched_movies_str)
            if not isinstance(watched_list, list): watched_list = []
        except:
            watched_list = [m.strip().strip("'") for m in watched_movies_str.strip('[]').split(',') if m.strip()]
        
        watched_list = [str(item) for item in watched_list if str(item).strip()]
            
        watched_df = df_movies[df_movies['Tên phim'].isin(watched_list)]
        
        # Kết hợp phim đã xem và phim được đề xuất
        combined_df = pd.concat([watched_df, recommended_movies_df], ignore_index=True)
        title = f"Độ Phổ Biến Thể Loại (Hồ sơ {st.session_state['logged_in_user']} & Đề xuất)"

    else:
        movie_row = df_movies[df_movies['Tên phim'].str.lower() == movie_name.lower()]
        if movie_row.empty: 
            st.error(f"Không tìm thấy thông tin phim gốc '{movie_name}' để so sánh.")
            return
        # Kết hợp phim gốc và phim được đề xuất
        combined_df = pd.concat([movie_row, recommended_movies_df], ignore_index=True)
        title = f"Độ Phổ Biến TB của Các Thể Loại Phim Liên Quan đến '{movie_name}'"

    genres_data = []
    # Chỉ làm việc với các cột cần thiết, đảm bảo không có NaN trong 'Độ phổ biến'
    combined_df = combined_df[['Thể loại phim', 'Độ phổ biến']].dropna()
    
    for index, row in combined_df.iterrows():
        # Phân tách chuỗi thể loại, đảm bảo mỗi thể loại được đếm riêng lẻ
        genres_list = [g.strip() for g in row['Thể loại phim'].split(',') if g.strip()]
        for genre in genres_list:
            genres_data.append({
                'Thể loại': genre,
                'Độ phổ biến': row['Độ phổ biến']
            })

    df_plot = pd.DataFrame(genres_data)
    
    if df_plot.empty:
        st.warning("Không đủ dữ liệu thể loại (Thường do thông tin phim bị thiếu thể loại) để vẽ biểu đồ. Vui lòng kiểm tra file `movie_info_1000.csv`.")
        return
        
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
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 1), ha='center', fontsize=10, weight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Thể loại phim")
    ax.set_ylabel("Độ Phổ Biến Trung Bình (Popularity Score)")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig) # Hiển thị trên Streamlit

# ==============================================================================
# IV. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================

def main_page(df_movies, cosine_sim):
    # Lấy dữ liệu người dùng từ Session State
    df_users = st.session_state['df_users']
    
    st.title(f"🎬 Chào mừng, {st.session_state['logged_in_user']}!")
    
    st.sidebar.title("Menu Đề Xuất")
    
    menu_choice = st.sidebar.radio(
        "Chọn chức năng:",
        ('Đề xuất theo Tên Phim', 'Đề xuất theo Hồ Sơ', 'Đăng Xuất')
    )

    # SỬ DỤNG on_click cho nút ĐĂNG XUẤT
    if st.sidebar.button("Đăng Xuất", key="logout_btn", on_click=logout):
        pass # Logic đã chuyển sang hàm logout()
        
    st.sidebar.write("-" * 20)

    if menu_choice == 'Đề xuất theo Tên Phim':
        st.header("1️⃣ Đề xuất dựa trên Nội dung (TF-IDF)")
        
        movie_titles_list = get_unique_movie_titles(df_movies)
        
        # Gán giá trị mặc định cho selectbox từ kết quả lần trước (nếu có)
        default_movie_name = st.session_state['last_sim_movie'] if st.session_state['last_sim_movie'] in movie_titles_list else movie_titles_list[0]
        movie_name = st.selectbox("🎥 Chọn tên phim bạn yêu thích:", options=movie_titles_list, index=movie_titles_list.index(default_movie_name))
        
        weight_sim = st.slider("⚖️ Trọng số Độ giống (Similarity)", 0.0, 1.0, 0.7, 0.1)
        weight_pop = 1 - weight_sim
        
        if st.button("Tìm Đề Xuất", key="find_sim"):
            result = recommend_movies_smart(movie_name, weight_sim, weight_pop, df_movies, cosine_sim)
            
            if not result.empty:
                # LƯU KẾT QUẢ VÀ TRẠNG THÁI VÀO SESSION STATE
                st.session_state['last_sim_result'] = result
                st.session_state['last_sim_movie'] = movie_name
                st.session_state['show_sim_plot'] = True # Tự động hiển thị biểu đồ
            else:
                # XÓA KẾT QUẢ CŨ VÀ ĐẶT LẠI TRẠNG THÁI KHI KHÔNG TÌM THẤY
                st.session_state['last_sim_result'] = pd.DataFrame()
                st.session_state['show_sim_plot'] = False
                st.warning("⚠️ Không tìm thấy đề xuất hoặc phim gốc không tồn tại.")
            st.rerun() # Chạy lại để hiển thị kết quả

        # --- HIỂN THỊ KẾT QUẢ VÀ BIỂU ĐỒ (ngoài khối if button) ---
        if not st.session_state['last_sim_result'].empty:
            result = st.session_state['last_sim_result']
            movie_name_for_display = st.session_state['last_sim_movie']

            st.subheader(f"🎬 10 Đề xuất phim dựa trên '{movie_name_for_display}':")
            st.dataframe(result, use_container_width=True)

            # Checkbox có giá trị mặc định được lưu trong Session State
            show_plot = st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", 
                                    value=st.session_state['show_sim_plot'], 
                                    key="plot_sim_check")

            if show_plot:
                recommended_movies_info = df_movies[df_movies['Tên phim'].isin(result['Tên phim'].tolist())]
                plot_genre_popularity(movie_name_for_display, recommended_movies_info, df_movies, is_user_based=False)

    elif menu_choice == 'Đề xuất theo Hồ Sơ':
        st.header("2️⃣ Đề xuất dựa trên Hồ sơ Người dùng")
        
        username = st.session_state['logged_in_user']
        user_row = df_users[df_users['Tên người dùng'] == username]
        
        if user_row.empty:
            st.error("LỖI: Không tìm thấy hồ sơ người dùng trong phiên. Vui lòng đăng ký lại.")
            return

        # Hiển thị 5 phim đã xem gần nhất
        recent_films_str = user_row['5 phim coi gần nhất'].iloc[0]
        recent_films = []
        try:
            recent_films = ast.literal_eval(recent_films_str)
            if not isinstance(recent_films, list): recent_films = []
        except:
            recent_films = [m.strip().strip("'") for m in recent_films_str.strip('[]').split(',') if m.strip()]
        
        recent_films_display = ', '.join([str(item) for item in recent_films if str(item).strip()])

        st.info(f"5 Phim đã xem gần nhất: {recent_films_display}")
        
        if st.button("Tìm Đề Xuất Hồ Sơ", key="find_profile"):
            recommendations = get_recommendations(username, df_movies, num_recommendations=10)

            if not recommendations.empty:
                # LƯU KẾT QUẢ VÀ TRẠNG THÁI VÀO SESSION STATE
                st.session_state['last_profile_recommendations'] = recommendations
                st.session_state['show_profile_plot'] = True # Tự động hiển thị biểu đồ
            else:
                # XÓA KẾT QUẢ CŨ VÀ ĐẶT LẠI TRẠNG THÁI KHI KHÔNG TÌM THẤY
                st.session_state['last_profile_recommendations'] = pd.DataFrame()
                st.session_state['show_profile_plot'] = False
                st.warning("⚠️ Không có đề xuất nào được tạo. Kiểm tra dữ liệu thể loại phim đã xem.")
            st.rerun() # Chạy lại để hiển thị kết quả

        # --- HIỂN THỊ KẾT QUẢ VÀ BIỂU ĐỒ (ngoài khối if button) ---
        if not st.session_state['last_profile_recommendations'].empty:
            recommendations = st.session_state['last_profile_recommendations']

            st.subheader(f"✅ 10 Đề xuất Phim Dành Cho Bạn:")
            st.dataframe(recommendations, use_container_width=True)
            
            # Checkbox có giá trị mặc định được lưu trong Session State
            show_plot_profile = st.checkbox("📊 Hiển thị Biểu đồ so sánh Thể loại", 
                                            value=st.session_state['show_profile_plot'],
                                            key="plot_profile_check")
            
            if show_plot_profile:
                recommended_movies_info = df_movies[df_movies['Tên phim'].isin(recommendations['Tên phim'].tolist())]
                plot_genre_popularity(None, 
                                      recommended_movies_info, 
                                      df_movies, is_user_based=True)


# ==============================================================================
# V. CHẠY ỨNG DỤNG CHÍNH
# ==============================================================================

if __name__ == '__main__':
    # 1. Tải dữ liệu tĩnh (Chỉ chạy 1 lần)
    df_movies, cosine_sim = load_and_preprocess_static_data()
    
    # 2. Khởi tạo dữ liệu người dùng (Sẽ được cập nhật khi đăng ký)
    initialize_user_data()
    
    # 3. Định tuyến trang
    if st.session_state['logged_in_user']:
        main_page(df_movies, cosine_sim)
    else:
        authentication_page(df_movies)
