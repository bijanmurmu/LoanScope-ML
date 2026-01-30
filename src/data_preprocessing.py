import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Store original column info for later use
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]
    
    # Store label encoders for each categorical column
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_columns, scaler, label_encoders, target_column


def preprocess_input(user_input, feature_columns, scaler, label_encoders):
    """Preprocess user input for prediction"""
    processed = []
    for col, value in zip(feature_columns, user_input):
        if col in label_encoders:
            try:
                processed.append(label_encoders[col].transform([value])[0])
            except ValueError:
                # Unknown category - use 0
                processed.append(0)
        else:
            processed.append(float(value))
    
    # Create DataFrame with feature names to avoid warning
    input_df = pd.DataFrame([processed], columns=feature_columns)
    processed = scaler.transform(input_df)
    
    return processed