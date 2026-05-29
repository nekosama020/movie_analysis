import pandas as pd

def clean_data(df_raw, drop_duplicates=True):
    """
    Hàm thực hiện tiền xử lý và làm sạch dữ liệu.
    """
    # Tạo bản sao để không làm ảnh hưởng dữ liệu gốc
    df = df_raw.copy()
    
    # Lưu lại thông tin thống kê trước khi làm sạch
    rows_before = len(df)
    missing_before = df.isnull().sum().sum()
    
    # 1. Xử lý giá trị thiếu (Missing values)
    if 'metascore' in df.columns:
        df['metascore'] = df['metascore'].fillna(df['metascore'].median())
    if 'rating' in df.columns:
        df['rating'] = df['rating'].fillna(df['rating'].median())
        
    # 2. Xóa dữ liệu trùng lặp (Duplicates)
    duplicates_removed = 0
    if drop_duplicates:
        before_drop = len(df)
        df = df.drop_duplicates(subset=['title', 'release_year', 'studio'], keep='first')
        duplicates_removed = before_drop - len(df)
        
    # 3. Tạo đặc trưng mới (Feature Engineering)
    if 'revenue' in df.columns and 'budget' in df.columns:
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = df['revenue'] / df['budget']
        
    if 'genres' in df.columns:
        df['primary_genre'] = df['genres'].apply(lambda x: str(x).split(',')[0].strip())
        
    # Thống kê sau khi làm sạch
    rows_after = len(df)
    missing_after = df.isnull().sum().sum()
    
    # Gom các chỉ số thống kê lại để gửi ra Dashboard
    stats = {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'missing_before': missing_before,
        'missing_after': missing_after,
        'duplicates_removed': duplicates_removed
    }
    
    return df, stats