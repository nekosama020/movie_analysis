import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_movie_models(df, target_col):
    """
    Huấn luyện và đánh giá các mô hình hồi quy để dự đoán target_col.
    """
    # 1. Xác định đặc trưng (Features)
    numeric_features = ['budget', 'runtime', 'release_year']
    categorical_features = ['primary_genre', 'studio', 'language']
    
    # Bổ sung các biến số khác nếu chúng không phải là biến mục tiêu
    if target_col != 'vote_count' and 'vote_count' in df.columns:
        numeric_features.append('vote_count')
    if target_col != 'metascore' and 'metascore' in df.columns:
        numeric_features.append('metascore')
        
    features = numeric_features + categorical_features
    
    # Lọc bỏ các dòng bị thiếu biến mục tiêu (nếu có)
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features]
    y = df_clean[target_col]
    
    # 2. Xây dựng Pipeline tiền xử lý cho Machine Learning
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 3. Khởi tạo các mô hình
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    # 4. Chia tập dữ liệu Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_pipeline = None
    
    # 5. Huấn luyện và Đánh giá
    for name, model in models.items():
        # Tạo pipeline hoàn chỉnh: Tiền xử lý -> Huấn luyện
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Huấn luyện mô hình
        clf.fit(X_train, y_train)
        
        # Dự đoán trên tập test
        y_pred = clf.predict(X_test)
        
        # Tính toán các chỉ số
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Lưu kết quả
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        })
        
        # Cập nhật mô hình tốt nhất
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = clf
            
    # Tạo DataFrame bảng xếp hạng mô hình
    leaderboard = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False).reset_index(drop=True)
    
    return leaderboard, best_pipeline, best_model_name, numeric_features, categorical_features