"""
feature_engineering.py
----------------------
Handles data loading and performs feature engineering on hotel booking data.
"""

import numpy as np
import pandas as pd

def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def engineer_customer_features(df):
    """Create engineered features such as total nights, ratios, and seasonal indicators."""
    df = df.drop(columns=['Booking_ID'], errors='ignore')
    df['total_people'] = df['no_of_adults'] + df['no_of_children']
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

    df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    df['arrival_month'] = df['arrival_date'].dt.month
    df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
    df['is_peak_day'] = df['arrival_day_of_week'].isin([4, 5, 6]).astype(int)
    df['is_peak_season'] = df['arrival_month'].isin([6, 7, 8, 12]).astype(int)

    df['cancel_ratio'] = df['no_of_previous_cancellations'] / (
        df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled'] + 1
    )
    df['price_per_night'] = np.where(
        df['total_nights'] == 0, 0, df['avg_price_per_room'] / df['total_nights']
    )
    return df
