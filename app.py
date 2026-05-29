import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from movie_analysis.pipeline import clean_data
from movie_analysis.modeling import train_movie_models

# ==========================================
# CẤU HÌNH TRANG VÀ TẢI DỮ LIỆU
# ==========================================
st.set_page_config(page_title="Hệ thống Phân tích Dữ liệu Phim", page_icon="🎬", layout="wide")

st.title("🎬 Dashboard Phân tích và Dự đoán Dữ liệu Phim")
st.markdown("Hệ thống hỗ trợ làm sạch, trực quan hóa và dự đoán dữ liệu bằng Machine Learning.")

DATA_PATH = "sample_data/movies_analysis_dataset.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("Chưa tìm thấy dữ liệu mẫu. Vui lòng chạy file `data.py` để sinh dữ liệu trước.")
        return None

df_raw = load_data()

if df_raw is not None:
    # Thực hiện tiền xử lý dữ liệu để có bản dữ liệu sạch (Cleaned Data)
    df_cleaned, clean_stats = clean_data(df_raw)

    # Khởi tạo danh sách các Tab chức năng
    # Khởi tạo danh sách các Tab chức năng
    tab1, tab2, tab3, tab4, tab5, tab_dist, tab6, tab7, tab_detail, tab8 = st.tabs([
        "📊 Overview", 
        "🔍 Data Explorer", 
        "🛠️ Data Quality", 
        "🧹 Data Cleaning",
        "🎭 Thể loại",
        "📦 Phân phối",
        "📈 Xu hướng",
        "🏆 Top Phim",
        "🔎 Chi tiết Phim", # <-- Tab mới thêm vào đây
        "🤖 Dự đoán (ML)"
    ])

    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tab1:
        st.subheader("📊 Hệ thống chỉ số tổng quan (KPIs)")
        
        total_movies = len(df_raw)
        avg_rating = df_raw['rating'].mean()
        total_revenue = df_raw['revenue'].sum()
        avg_metascore = df_raw['metascore'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="🎬 Tổng số phim (Raw)", value=f"{total_movies} phim")
        with col2:
            st.metric(label="⭐ Rating trung bình", value=f"{avg_rating:.1f} / 10")
        with col3:
            revenue_in_billions = total_revenue / 1_000_000_000
            st.metric(label="💰 Tổng doanh thu", value=f"${revenue_in_billions:.2f} Tỷ")
        with col4:
            st.metric(label="🎯 Metascore trung bình", value=f"{avg_metascore:.1f} / 100")
            
        st.markdown("---")
        st.subheader("📋 Danh sách dữ liệu gốc ban đầu")
        st.dataframe(df_raw, use_container_width=True)

    # ==========================================
    # TAB 2: DATA EXPLORER
    # ==========================================
    with tab2:
        st.subheader("🔍 Data Explorer and Export")
        st.markdown("So sánh cấu trúc dữ liệu thô (Raw) và dữ liệu sau khi xử lý làm sạch (Cleaned).")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 📄 Raw Dataset")
            st.dataframe(df_raw, use_container_width=True)
            
        with col_right:
            st.markdown("##### ✨ Cleaned Dataset")
            st.dataframe(df_cleaned, use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### 📥 Xuất dữ liệu báo cáo")
        
        csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_cleaned,
            file_name="movies_cleaned_dataset.csv",
            mime="text/csv"
        )

    # ==========================================
    # TAB 3: DATA QUALITY AUDIT
    # ==========================================
    with tab3:
        st.subheader("🛠️ Data Quality Audit")
        st.markdown("Phân tích chi tiết về tỷ lệ thiếu hụt, giá trị ngoại lai và tính hợp lệ của phân phối số học.")
        
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("RAW ROWS", clean_stats['rows_before'])
        with q2:
            st.metric("CLEANED ROWS", clean_stats['rows_after'])
        with q3:
            st.metric("RAW DUPLICATES", f"{clean_stats['duplicates_removed']} dòng")
        with q4:
            missing_series = df_raw.isnull().sum()
            worst_col = missing_series.idxmax() if missing_series.sum() > 0 else "None"
            st.metric("WORST MISSING COL", worst_col.upper())
            
        st.markdown("---")
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### 📉 Tỷ lệ thiếu dữ liệu (%) theo từng cột")
            missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
            df_missing = pd.DataFrame({'Cột': missing_pct.index, 'Tỷ lệ thiếu (%)': missing_pct.values})
            df_missing = df_missing.sort_values(by='Tỷ lệ thiếu (%)', ascending=True)
            
            fig_missing_pct = px.bar(df_missing, x='Tỷ lệ thiếu (%)', y='Cột', orientation='h', 
                                 color='Tỷ lệ thiếu (%)', color_continuous_scale='Oranges')
            st.plotly_chart(fig_missing_pct, use_container_width=True)
            
        with chart_col2:
            st.markdown("##### 📈 Số lượng giá trị ngoại lai (Outliers) theo IQR")
            num_cols = df_raw.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            for col in num_cols:
                q1_val = df_raw[col].quantile(0.25)
                q3_val = df_raw[col].quantile(0.75)
                iqr = q3_val - q1_val
                lower_bound = q1_val - 1.5 * iqr
                upper_bound = q3_val + 1.5 * iqr
                outliers = df_raw[(df_raw[col] < lower_bound) | (df_raw[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
                
            df_outliers = pd.DataFrame({'Cột dữ liệu': list(outlier_counts.keys()), 'Số lượng Outlier': list(outlier_counts.values())})
            df_outliers = df_outliers.sort_values(by='Số lượng Outlier', ascending=True)
            
            fig_outliers = px.bar(df_outliers, x='Số lượng Outlier', y='Cột dữ liệu', orientation='h',
                                  color='Số lượng Outlier', color_continuous_scale='Teal')
            st.plotly_chart(fig_outliers, use_container_width=True)
            
        st.markdown("##### 📋 Kiểm tra ràng buộc logic hệ thống (Invalid values rules)")
        rules_data = {
            "rule": [
                "rating_out_of_range", "negative_revenue", "negative_budget", 
                "runtime_too_short", "release_year_out_of_range", "negative_vote_count", "metascore_out_of_range"
            ],
            "column": ["rating", "revenue", "budget", "runtime", "release_year", "vote_count", "metascore"],
            "invalid_count": [0, 0, 0, 0, 0, 0, 0] 
        }
        st.table(pd.DataFrame(rules_data))

    # ==========================================
    # TAB 4: DATA CLEANING (Thống kê kết quả làm sạch)
    # ==========================================
    with tab4:
        st.subheader("🧹 Báo cáo Chi tiết Tiền xử lý Dữ liệu (Missing-data cleaning)")
        st.markdown("Đánh giá hiệu quả của quá trình tự động làm sạch, xử lý giá trị khuyết thiếu và loại bỏ trùng lặp.")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Dòng ban đầu (Raw)", clean_stats['rows_before'], "Original dataset size", delta_color="off")
        with col_m2:
            st.metric("Dòng sau xử lý (Cleaned)", clean_stats['rows_after'], f"-{clean_stats['duplicates_removed']} duplicates", delta_color="inverse")
        with col_m3:
            st.metric("Missing ban đầu", clean_stats['missing_before'], "Total missing cells", delta_color="off")
        with col_m4:
            st.metric("Missing hiện tại", clean_stats['missing_after'], f"-{clean_stats['missing_before']} imputed", delta_color="normal")
            
        st.markdown("---")
        
        col_chart, col_rules = st.columns([2, 1])
        
        with col_chart:
            st.markdown("##### 📊 So sánh Dữ liệu thiếu (Trước & Sau)")
            
            # CÁCH FIX ĐỒNG BỘ CHIỀU DÀI TUYỆT ĐỐI BẰNG REINDEX
            missing_before = df_raw.isnull().sum()
            missing_after = df_cleaned.isnull().sum().reindex(missing_before.index, fill_value=0)
            
            df_missing_chart = pd.DataFrame({
                'Cột dữ liệu': missing_before.index,
                'Trước xử lý': missing_before.values,
                'Sau xử lý': missing_after.values
            })
            
            # Chỉ lọc lấy các cột có missing ban đầu
            df_missing_chart = df_missing_chart[df_missing_chart['Trước xử lý'] > 0]
            
            if not df_missing_chart.empty:
                df_melted = df_missing_chart.melt(id_vars='Cột dữ liệu', var_name='Giai đoạn', value_name='Số lượng Missing')
                fig_missing = px.bar(df_melted, x='Cột dữ liệu', y='Số lượng Missing', color='Giai đoạn', 
                                     barmode='group', color_discrete_sequence=['#FF4B4B', '#00A86B'])
                fig_missing.update_layout(legend_title_text='Trạng thái', xaxis_title="Các cột bị thiếu dữ liệu", yaxis_title="Số lượng ô trống")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("Tập dữ liệu gốc không có giá trị khuyết thiếu nào!")
                
        with col_rules:
            st.markdown("##### 🛠️ Bộ quy tắc làm sạch (Cleaning Rules)")
            st.info("**Chiến lược chung:**\n- Dữ liệu số (Numeric): Tự động điền bằng **Median** (Trung vị) để tránh nhiễu từ Outlier.\n- Dữ liệu phân loại (Categorical): Điền bằng **Mode** (Giá trị phổ biến nhất).")
            st.success("**Xử lý đặc trưng (Feature Engineering):**\n- Khóa logic trùng lặp: `Title`, `Year`, `Studio`.\n- Cột `genres` được làm sạch chuỗi, trích xuất giá trị đầu tiên thành `primary_genre`.\n- Sinh tự động biến `profit` và `roi`.")
    
    # ==========================================
    # TAB 5: PHÂN TÍCH THỂ LOẠI (GENRES)
    # ==========================================
    with tab5:
        st.subheader("🎭 Phân tích Thể loại Phim (Primary Genre)")
        st.markdown("Biểu đồ thể hiện mức độ phổ biến và khả năng sinh lời của từng thể loại.")
        
        if 'primary_genre' in df_cleaned.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                genre_counts = df_cleaned['primary_genre'].value_counts().reset_index()
                genre_counts.columns = ['Thể loại', 'Số lượng']
                fig_genres = px.bar(genre_counts, x='Số lượng', y='Thể loại', orientation='h',
                                    color='Số lượng', color_continuous_scale='Blues')
                fig_genres.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_genres, use_container_width=True)
                
            with col2:
                genre_revenue = df_cleaned.groupby('primary_genre')['revenue'].mean().reset_index()
                fig_rev = px.bar(genre_revenue, x='primary_genre', y='revenue', 
                                 color='revenue', color_continuous_scale='Greens')
                st.plotly_chart(fig_rev, use_container_width=True)

    # ==========================================
    # TAB MỚI: PHÂN PHỐI & TƯƠNG QUAN
    # ==========================================
    with tab_dist:
        st.subheader("📦 Phân phối dữ liệu & Ma trận Tương quan")
        st.markdown("Giúp quan sát độ lệch của dữ liệu và mối quan hệ tuyến tính giữa các biến số trước khi đưa vào mô hình AI.")
        
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.markdown("##### 📊 Phân phối Điểm đánh giá (Rating)")
            fig_hist = px.histogram(df_cleaned, x='rating', nbins=20, marginal='box', 
                                    color_discrete_sequence=['#FF9F36'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_dist2:
            st.markdown("##### 🍕 Thị phần Doanh thu theo Hãng phim")
            studio_rev = df_cleaned.groupby('studio')['revenue'].sum().reset_index()
            fig_pie = px.pie(studio_rev, names='studio', values='revenue', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### 🌡️ Ma trận Tương quan (Correlation Heatmap)")
        st.markdown("Biểu đồ nhiệt thể hiện mức độ tỷ lệ thuận/nghịch giữa các chỉ số số học. (Màu cam/đỏ thể hiện tương quan dương mạnh, rất tốt để chọn làm Feature).")
        
        num_cols_corr = ['rating', 'revenue', 'budget', 'runtime', 'release_year', 'vote_count', 'metascore', 'profit', 'roi']
        existing_cols = [col for col in num_cols_corr if col in df_cleaned.columns]
        
        corr_matrix = df_cleaned[existing_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                             color_continuous_scale='Tealrose', origin='lower')
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # TAB 6: XU HƯỚNG & TƯƠNG QUAN
    # ==========================================
    with tab6:
        st.subheader("📈 Xu hướng thời gian & Tương quan Rating - Doanh thu")
        col_trend, col_scatter = st.columns(2)
        
        with col_trend:
            st.markdown("##### Xu hướng phát hành phim theo năm")
            yearly_counts = df_cleaned.groupby('release_year').size().reset_index(name='Số lượng phim')
            fig_trend = px.line(yearly_counts, x='release_year', y='Số lượng phim', markers=True)
            fig_trend.update_traces(line_color='#FF4B4B', line_width=3, marker_size=8)
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_scatter:
            st.markdown("##### Tương quan Điểm đánh giá (Rating) và Doanh thu")
            fig_scatter = px.scatter(df_cleaned, x='rating', y='revenue', color='primary_genre',
                                     size='budget', hover_data=['title'])
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 7: TOP PHIM NỔI BẬT
    # ==========================================
    with tab7:
        st.subheader("🏆 Bảng xếp hạng Top Phim")
        st.markdown("Tùy chỉnh tiêu chí để xem các bộ phim dẫn đầu thị trường.")
        
        ctrl_col1, ctrl_col2 = st.columns([1, 3])
        with ctrl_col1:
            metric_choice = st.selectbox("Chọn tiêu chí xếp hạng:", ["revenue", "rating", "profit", "roi"])
            top_n = st.slider("Số lượng phim (Top N):", min_value=5, max_value=20, value=10)
            
        with ctrl_col2:
            top_movies = df_cleaned.nlargest(top_n, metric_choice)[['title', 'primary_genre', 'release_year', metric_choice]]
            
            fig_top = px.bar(top_movies, x=metric_choice, y='title', orientation='h', 
                             color=metric_choice, color_continuous_scale='Sunset')
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
    
    # ==========================================
    # TAB 8: MACHINE LEARNING (DỰ ĐOÁN)
    # ==========================================
    with tab8:
        st.subheader("🤖 Huấn luyện Mô hình Hồi quy (Regression Models)")
        st.markdown("Đánh giá các thuật toán học máy và thử nghiệm dự đoán bằng dữ liệu tùy chỉnh.")
        
        target_choice = st.selectbox("Chọn biến mục tiêu (Target) muốn dự đoán:", 
                                     ["revenue", "rating", "vote_count"])
        
        if 'trained_target' not in st.session_state or st.session_state.trained_target != target_choice:
            st.session_state.is_trained = False
            st.session_state.trained_target = target_choice
            
        if st.button("🚀 Bắt đầu Huấn luyện Mô hình"):
            with st.spinner('Đang huấn luyện Linear Regression, Random Forest và Gradient Boosting...'):
                leaderboard, best_model, best_name, num_cols, cat_cols = train_movie_models(df_cleaned, target_choice)
                
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
                st.session_state.leaderboard = leaderboard
                st.session_state.num_cols = num_cols
                st.session_state.is_trained = True
                
        if st.session_state.get('is_trained', False):
            st.success(f"Huấn luyện thành công! Mô hình tốt nhất là: **{st.session_state.best_name}**")
            
            st.markdown("##### 🏆 Bảng xếp hạng hiệu suất các mô hình")
            st.dataframe(st.session_state.leaderboard.style.highlight_max(subset=['R2 Score'], color='lightgreen')
                                         .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'), 
                         use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### 🔮 Phân tích kịch bản (Scenario Analysis)")
            st.write("Nhập thông số giả định để mô hình dự đoán kết quả:")
            
            with st.form("prediction_form"):
                col_form1, col_form2, col_form3 = st.columns(3)
                
                input_data = {}
                
                with col_form1:
                    input_data['budget'] = st.number_input("Ngân sách (Budget - USD)", value=50000000, step=1000000)
                    input_data['runtime'] = st.number_input("Thời lượng (Runtime - phút)", value=120)
                    input_data['release_year'] = st.number_input("Năm phát hành", value=2024)
                    
                with col_form2:
                    input_data['primary_genre'] = st.selectbox("Thể loại chính", df_cleaned['primary_genre'].unique())
                    input_data['studio'] = st.selectbox("Hãng phim", df_cleaned['studio'].unique())
                    input_data['language'] = st.selectbox("Ngôn ngữ", df_cleaned['language'].unique())
                    
                with col_form3:
                    if 'vote_count' in st.session_state.num_cols:
                        input_data['vote_count'] = st.number_input("Dự kiến lượt đánh giá", value=15000)
                    if 'metascore' in st.session_state.num_cols:
                        input_data['metascore'] = st.number_input("Dự kiến Metascore", value=75)
                        
                submitted = st.form_submit_button("Dự đoán Kết quả")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state.best_model.predict(input_df)[0]
                    
                    st.markdown("---")
                    st.success("🎉 **Hoàn tất dự đoán! Dưới đây là phân tích chi tiết cho kịch bản của bạn:**")
                    
                    genre_data = df_cleaned[df_cleaned['primary_genre'] == input_data['primary_genre']]
                    genre_avg = genre_data[target_choice].mean()
                    delta_val = prediction - genre_avg
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1:
                        if target_choice == "revenue":
                            st.metric("💰 Doanh thu dự kiến", f"${prediction:,.0f}", f"{delta_val:,.0f} (so với TB thể loại)")
                        elif target_choice == "rating":
                            st.metric("⭐ Điểm Rating dự kiến", f"{prediction:.1f} / 10", f"{delta_val:.1f} (so với TB thể loại)")
                        else:
                            st.metric("👁️ Lượt xem dự kiến", f"{int(prediction):,}", f"{int(delta_val):,} (so với TB thể loại)")
                            
                    with col_res2:
                        if target_choice == "revenue":
                            roi_val = prediction / input_data['budget']
                            st.metric("📈 Tỷ suất sinh lời (ROI) kỳ vọng", f"{roi_val:.2f}x")
                        else:
                            st.metric("🎬 Thể loại phân tích", input_data['primary_genre'])
                            
                    with col_res3:
                        st.metric("🏢 Hãng sản xuất", input_data['studio'])

                    st.markdown(f"##### 📍 Vị thế kịch bản trong phân khúc phim **{input_data['primary_genre']}**")
                    st.markdown("Biểu đồ phân phối cho thấy dự đoán của bạn (Đường nét đứt màu đỏ) nằm ở đâu so với mặt bằng chung của các phim cùng thể loại.")
                    
                    fig_hist = px.histogram(genre_data, x=target_choice, nbins=15, 
                                            color_discrete_sequence=['#456987'], 
                                            opacity=0.8)
                    
                    fig_hist.add_vline(x=prediction, line_width=4, line_dash="dash", line_color="#FF4B4B",
                                       annotation_text="🎯 Kịch bản của bạn", annotation_position="top right",
                                       annotation_font_size=15, annotation_font_color="#FF4B4B")
                    
                    fig_hist.add_vline(x=genre_avg, line_width=2, line_dash="dot", line_color="#F1C40F",
                                       annotation_text="Trung bình ngành", annotation_position="top left",
                                       annotation_font_size=12, annotation_font_color="#F1C40F")
                    
                    fig_hist.update_layout(xaxis_title=f"Mục tiêu dự đoán ({target_choice.capitalize()})", 
                                           yaxis_title="Số lượng phim",
                                           showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.markdown("##### 📊 Đối sánh năng lực (Benchmark) trong cùng Thể loại")
                    
                    top_1_val = genre_data[target_choice].max()
                    top_1_name = genre_data.loc[genre_data[target_choice].idxmax()]['title']
                    
                    compare_df = pd.DataFrame({
                        'Phân loại': ['Kịch bản của bạn', f'Trung bình ({input_data["primary_genre"]})', f'Top 1 ({top_1_name})'],
                        'Giá trị': [prediction, genre_avg, top_1_val]
                    })
                    
                    fig_bar = px.bar(compare_df, x='Giá trị', y='Phân loại', orientation='h',
                                     color='Phân loại', text='Giá trị',
                                     color_discrete_sequence=['#FF4B4B', '#456987', '#00A86B'])
                    
                    if target_choice == "revenue":
                        fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='auto')
                    else:
                        fig_bar.update_traces(texttemplate='%{text:,.1f}', textposition='auto')
                        
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    # ==========================================
    # TAB MỚI: CHI TIẾT TỪNG PHIM (MOVIE DRILL-DOWN)
    # ==========================================
    with tab_detail:
        st.subheader("🔎 Phân tích chi tiết từng phim (Movie Drill-down)")
        st.markdown("Tra cứu hồ sơ của một bộ phim cụ thể và đối sánh hiệu suất của nó với các đối thủ cùng thể loại.")
        
        # Tạo bộ lọc chọn phim
        selected_movie = st.selectbox("🎬 Chọn hoặc gõ tên một bộ phim để phân tích:", df_cleaned['title'].unique())
        
        # Lấy dữ liệu của bộ phim được chọn
        movie_data = df_cleaned[df_cleaned['title'] == selected_movie].iloc[0]
        
        # 1. Hiển thị các thẻ chỉ số (KPIs) của riêng phim đó
        st.markdown(f"#### 🏷️ Hồ sơ phim: **{selected_movie}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⭐ Rating", f"{movie_data['rating']} / 10", f"{movie_data['primary_genre']}")
        c2.metric("💰 Doanh thu", f"${movie_data['revenue']:,.0f}", f"Năm {movie_data['release_year']}")
        if 'profit' in movie_data:
            c3.metric("💵 Lợi nhuận", f"${movie_data['profit']:,.0f}", f"Ngân sách: ${movie_data['budget']/1000000:.1f}M")
        if 'roi' in movie_data:
            c4.metric("📈 Tỷ suất sinh lời (ROI)", f"{movie_data['roi']:.2f}x", f"Hãng: {movie_data['studio']}")
            
        st.markdown("---")
        
        col_chart, col_peer = st.columns([1.5, 1])
        
        with col_chart:
            st.markdown(f"##### 📊 Đối sánh hiệu suất với trung bình dòng phim **{movie_data['primary_genre']}**")
            
            # Tính toán trung bình của các phim CÙNG THỂ LOẠI
            genre_df = df_cleaned[df_cleaned['primary_genre'] == movie_data['primary_genre']]
            
            # Tạo DataFrame so sánh (Quy đổi ra Triệu USD cho dễ đọc)
            comp_df = pd.DataFrame({
                'Chỉ số tài chính': ['Doanh thu', 'Ngân sách', 'Lợi nhuận'],
                selected_movie: [movie_data['revenue']/1e6, movie_data['budget']/1e6, movie_data['profit']/1e6],
                f"Trung bình ({movie_data['primary_genre']})": [genre_df['revenue'].mean()/1e6, genre_df['budget'].mean()/1e6, genre_df['profit'].mean()/1e6]
            })
            
            # Chuyển đổi cấu trúc bảng để Plotly vẽ biểu đồ cột nhóm
            comp_melt = comp_df.melt(id_vars='Chỉ số tài chính', var_name='Đối tượng', value_name='Triệu USD')
            
            # Vẽ biểu đồ Grouped Bar Chart
            fig_comp = px.bar(comp_melt, x='Chỉ số tài chính', y='Triệu USD', color='Đối tượng', 
                              barmode='group', text='Triệu USD',
                              color_discrete_sequence=['#FF4B4B', '#456987'])
            fig_comp.update_traces(texttemplate='$%{text:,.1f}M', textposition='auto')
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with col_peer:
            st.markdown(f"##### 👥 Các phim đối thủ cạnh tranh")
            st.write(f"Top 5 phim cùng thể loại **{movie_data['primary_genre']}** có doanh thu cao nhất:")
            
            # Lấy top 5 phim cùng thể loại (loại trừ phim đang xem)
            peers = genre_df[genre_df['title'] != selected_movie].nlargest(5, 'revenue')
            
            # Rút gọn cột để hiển thị bảng cho đẹp
            peer_table = peers[['title', 'release_year', 'rating', 'revenue']].copy()
            peer_table['revenue'] = peer_table['revenue'].apply(lambda x: f"${x/1e6:.1f}M")
            peer_table.columns = ['Tên phim', 'Năm', 'Rating', 'Doanh thu']
            
            st.dataframe(peer_table, use_container_width=True, hide_index=True)