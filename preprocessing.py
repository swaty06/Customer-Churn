# preprocessing.py
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Categorical Encoding Function
# ------------------------------
def encode_dataset(df):
    """
    Encode categorical features - matches your Kaggle training code
    
    Parameters:
    df (pd.DataFrame): Raw input dataframe
    
    Returns:
    pd.DataFrame: Encoded dataframe
    """
    df_encoded = df.copy()
    
    # 1. Ordinal mapping
    subscription_order = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df_encoded['SubscriptionType'] = df_encoded['SubscriptionType'].map(subscription_order)
    
    # 2. Binary mapping (Yes/No → 1/0)
    binary_cols = ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled']
    for col in binary_cols:
        df_encoded[col] = df_encoded[col].map({'No': 0, 'Yes': 1})
    
    # 3. Normalize categorical values to match Kaggle format
    # This ensures the one-hot encoding column names match exactly
    if 'PaymentMethod' in df_encoded.columns:
        # Normalize to match Kaggle's exact format
        payment_mapping = {
            'Credit Card': 'Credit card',
            'Debit Card': 'Debit card', 
            'PayPal': 'Paypal',
            'Bank Transfer': 'Bank transfer',
            'Electronic check': 'Electronic check',
            'Mailed check': 'Mailed check'
        }
        df_encoded['PaymentMethod'] = df_encoded['PaymentMethod'].replace(payment_mapping)
    
    if 'Gender' in df_encoded.columns:
        # Normalize Gender
        gender_mapping = {'M': 'Male', 'F': 'Female', 'Other': 'Other'}
        df_encoded['Gender'] = df_encoded['Gender'].replace(gender_mapping)
    
    # 4. Nominal one-hot encoding
    nominal_cols = ['PaymentMethod', 'ContentType', 'DeviceRegistered', 'GenrePreference', 'Gender']
    existing_cols = [col for col in nominal_cols if col in df_encoded.columns]
    
    if existing_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=existing_cols, drop_first=True)
    
    # Convert to int (matching your Kaggle code)
    df_encoded = df_encoded.astype(int)
    
    return df_encoded

# ------------------------------
# Load saved scaler and training columns
# ------------------------------
def load_scaler(scaler_path='scaler.pkl'):
    """Load the saved fitted scaler"""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
        return scaler
    else:
        raise FileNotFoundError(f"❌ Scaler file not found at {scaler_path}")

def load_training_columns(columns_path='training_columns.pkl'):
    """Load the saved training column names"""
    if os.path.exists(columns_path):
        columns = joblib.load(columns_path)
        print(f"✅ Loaded {len(columns)} training columns from {columns_path}")
        return columns
    else:
        raise FileNotFoundError(f"❌ Training columns file not found at {columns_path}")

# Load the scaler and training columns globally
try:
    scaler = load_scaler()
    training_columns = load_training_columns()
    
    # Remove 'Churn' if it exists (target variable shouldn't be in features)
    if training_columns and 'Churn' in training_columns:
        training_columns = [col for col in training_columns if col != 'Churn']
        print(f"⚠️ Removed 'Churn' from training columns. Now have {len(training_columns)} features.")
    
except FileNotFoundError as e:
    print(f"Warning: {e}")
    scaler = None
    training_columns = None

# ------------------------------
# Feature list
# ------------------------------
features = ['AccountAge', 'TotalCharges', 'ViewingHoursPerWeek', 
            'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating', 
            'SupportTicketsPerMonth', 'WatchlistSize']

# ------------------------------
# Transform function for scaling
# ------------------------------
def transform_scaler(df):
    """
    Transform data using the loaded fitted scaler
    
    Parameters:
    df (pd.DataFrame): Encoded input dataframe
    
    Returns:
    pd.DataFrame: Scaled dataframe
    """
    if scaler is None:
        raise ValueError("Scaler not loaded. Please ensure 'scaler.pkl' exists.")
    
    df_scaled = df.copy()
    
    # Check if all required features exist
    missing_features = [f for f in features if f not in df_scaled.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    # Transform using the fitted scaler
    df_scaled[features] = scaler.transform(df_scaled[features])
    
    return df_scaled

# ------------------------------
# Complete preprocessing pipeline
# ------------------------------
def preprocess_input(input_df):
    """
    Complete preprocessing pipeline for a single prediction
    
    Parameters:
    input_df (pd.DataFrame): Raw input data
    
    Returns:
    pd.DataFrame: Fully preprocessed dataframe ready for prediction
    """
    if training_columns is None:
        raise ValueError("Training columns not loaded. Please ensure 'training_columns.pkl' exists.")
    
    # Step 1: Encode categorical features
    df_encoded = encode_dataset(input_df.copy())
    
    # DEBUG: Print columns after encoding
    print(f"\n🔍 DEBUG INFO:")
    print(f"Columns after encoding: {len(df_encoded.columns)}")
    print(f"Expected training columns: {len(training_columns)}")
    print(f"\nColumns in encoded data: {sorted(df_encoded.columns.tolist())}")
    print(f"\nExpected training columns: {sorted(training_columns)}")
    
    # Find differences
    extra_cols = set(df_encoded.columns) - set(training_columns)
    missing_cols = set(training_columns) - set(df_encoded.columns)
    
    if extra_cols:
        print(f"\n⚠️ Extra columns (will be removed): {extra_cols}")
    if missing_cols:
        print(f"\n⚠️ Missing columns (will be added with 0s): {missing_cols}")
    
    # Step 2: Align columns with training data BEFORE scaling
    # Add missing columns with 0s
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Remove extra columns and reorder to match training
    df_encoded = df_encoded[training_columns]
    
    print(f"\n✅ Final columns after alignment: {len(df_encoded.columns)}")
    
    # Step 3: Scale numerical features (now with correct number of columns)
    df_scaled = transform_scaler(df_encoded)
    
    return df_scaled