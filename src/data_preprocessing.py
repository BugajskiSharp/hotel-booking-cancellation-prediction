"""
data_preprocessing.py
---------------------
Handles data splitting, encoding, and tensor conversion.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def split_data(df, target_col='booking_status', test_size=0.2, val_size=0.2, random_state=123):
    """Split dataset into train/val/test while preserving class distribution."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_and_tensorize(X_train, X_val, X_test, y_train, y_val, y_test, device=torch.device('cpu')):
    """
    Full preprocessing pipeline:
    - Outlier clipping
    - One-hot encoding
    - Label encoding
    - Scaling
    - Conversion to tensors
    """
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

    # Outlier Clipping on Continuous Features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    binary_vars = ['required_car_parking_space', 'repeated_guest', 'is_peak_day', 
                   'is_peak_season', 'arrival_month']
    continuous_cols = [col for col in numeric_cols if col not in binary_vars]

    for col in continuous_cols:
        lower = X_train[col].quantile(0.005)
        upper = X_train[col].quantile(0.995)
        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col]   = X_val[col].clip(lower, upper)
        X_test[col]  = X_test[col].clip(lower, upper)

    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # One-Hot Encode Categorical Variables
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    def encode_data(X):
        encoded = pd.DataFrame(
            encoder.transform(X[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index
        )
        return pd.concat([X.drop(columns=categorical_cols), encoded], axis=1)

    X_train_enc = encode_data(X_train)
    X_val_enc = encode_data(X_val)
    X_test_enc = encode_data(X_test)

    # Convert Boolean Columns to int
    bool_cols = X_train_enc.select_dtypes(include='bool').columns
    for df in [X_train_enc, X_val_enc, X_test_enc]:
        df[bool_cols] = df[bool_cols].astype(int)

    # Cyclical Feature Encoding
    def add_cyclical_features(X):
        X = X.copy()
        X['arrival_month'] = X['arrival_date'].dt.month
        X['arrival_month_sin'] = np.sin(2 * np.pi * X['arrival_month'] / 12)
        X['arrival_month_cos'] = np.cos(2 * np.pi * X['arrival_month'] / 12)
        
        X['arrival_day_of_week'] = X['arrival_date'].dt.dayofweek
        X['arrival_day_sin'] = np.sin(2 * np.pi * X['arrival_day_of_week'] / 7)
        X['arrival_day_cos'] = np.cos(2 * np.pi * X['arrival_day_of_week'] / 7)
        
        X = X.drop(columns=['arrival_date', 'arrival_month', 'arrival_day_of_week'])
        return X

    X_train_enc = add_cyclical_features(X_train_enc)
    X_val_enc = add_cyclical_features(X_val_enc)
    X_test_enc = add_cyclical_features(X_test_enc)

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train_enc[continuous_cols])
    X_train_enc[continuous_cols] = scaler.transform(X_train_enc[continuous_cols])
    X_val_enc[continuous_cols] = scaler.transform(X_val_enc[continuous_cols])
    X_test_enc[continuous_cols] = scaler.transform(X_test_enc[continuous_cols])

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_enc.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_enc.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_enc.values, dtype=torch.float32).to(device)

    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val_enc, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long).to(device)

    return (X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_tensor, y_val_tensor, y_test_tensor,
            le, encoder, scaler)