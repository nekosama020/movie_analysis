import pandas as pd
import numpy as np
import os

def generate_sample_data(filepath="sample_data/movies_analysis_dataset.csv"):
    """
    Hàm tạo dữ liệu mẫu mô phỏng danh sách phim, bao gồm cả dữ liệu thiếu (missing values) 
    và dữ liệu bất thường (outliers) để phục vụ việc demo tính năng làm sạch.
    """
    # Đảm bảo thư mục sample_data tồn tại
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.random.seed(42)
    num_rows = 328 # Kích thước như trong tài liệu

    # 1. Tạo các cột dữ liệu cơ bản
    data = {
        'title': [f'Movie {i:03d}' for i in range(num_rows)],
        'genres': np.random.choice(
            ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Action, Sci-Fi', 'Drama, Romance'], 
            num_rows
        ),
        'rating': np.round(np.random.uniform(4.0, 9.5, num_rows), 1),
        'revenue': np.round(np.random.uniform(10_000_000, 1_000_000_000, num_rows), 2),
        'budget': np.round(np.random.uniform(5_000_000, 200_000_000, num_rows), 2),
        'runtime': np.random.randint(80, 180, num_rows),
        'release_year': np.random.randint(2000, 2025, num_rows),
        'vote_count': np.random.randint(100, 50000, num_rows),
        'metascore': np.random.randint(30, 100, num_rows),
        'studio': np.random.choice(['Warner Bros', 'Universal', 'Paramount', 'Disney', 'Sony'], num_rows),
        'language': np.random.choice(['English', 'Spanish', 'French', 'Korean'], num_rows)
    }

    df = pd.DataFrame(data)

    # 2. Chủ động chèn Missing Values (NaN) để test tính năng làm sạch (Cleaning)
    missing_indices_metascore = np.random.choice(df.index, 19, replace=False)
    df.loc[missing_indices_metascore, 'metascore'] = np.nan

    missing_indices_rating = np.random.choice(df.index, 17, replace=False)
    df.loc[missing_indices_rating, 'rating'] = np.nan

    # 3. Chủ động chèn Dữ liệu trùng lặp (Duplicates) (8 dòng như trong báo cáo)
    duplicates = df.sample(8)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Lưu ra file CSV
    df.to_csv(filepath, index=False)
    print(f"Đã tạo thành công dữ liệu mẫu tại: {filepath} với {len(df)} dòng.")
    
    return df

# Chạy đoạn code này nếu thực thi file data.py trực tiếp
if __name__ == "__main__":
    generate_sample_data()