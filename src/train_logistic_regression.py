from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data_preprocessing import load_and_preprocess_data

def train_logistic_regression(data_path):
    X, y, feature_columns, scaler, label_encoders, target_column = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Logistic Regression Accuracy:", round(accuracy * 100, 2), "%")
    
    return model, feature_columns, scaler, label_encoders, target_column

if __name__ == "__main__":
    train_logistic_regression("dataset/loan_approval.csv")