import argparse
from src.train_logistic_regression import train_logistic_regression
from src.train_decision_tree import train_decision_tree
from src.data_preprocessing import preprocess_input

def get_user_prediction(model, model_name, feature_columns, scaler, label_encoders):
    """Get user input and make prediction"""
    print(f"\n--- Predict using {model_name} ---")
    print("Enter values for each feature:")
    
    user_input = []
    for col in feature_columns:
        value = input(f"  {col}: ")
        user_input.append(value)
    
    # Preprocess and predict
    processed_input = preprocess_input(user_input, feature_columns, scaler, label_encoders)
    prediction = model.predict(processed_input)[0]
    
    # Convert prediction to readable format
    if prediction == 1 or prediction == True:
        result = "APPROVED ✓"
    else:
        result = "REJECTED ✗"
    
    print(f"\n  {model_name} Prediction: {result}")
    return prediction

def main():
    parser = argparse.ArgumentParser(description="LoanScope-ML: Loan Approval Prediction")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="dataset/loan_approval.csv",
        help="Path to the CSV dataset file (default: dataset/loan_approval.csv)"
    )
    parser.add_argument(
        "--predict", "-p",
        action="store_true",
        help="Enable interactive prediction mode after training"
    )
    args = parser.parse_args()

    print("===== LoanScope-ML: Loan Approval Prediction =====\n")
    print(f"Using dataset: {args.dataset}\n")

    print("Training Logistic Regression Model...")
    lr_model, feature_columns, scaler, label_encoders, target_column = train_logistic_regression(args.dataset)

    print("\nTraining Decision Tree Model...")
    dt_model, _, _, _, _ = train_decision_tree(args.dataset)

    print("\nModel training completed successfully.")
    
    # Interactive prediction mode
    if args.predict:
        while True:
            print("\n" + "="*50)
            print("Enter your data for loan prediction:")
            
            get_user_prediction(lr_model, "Logistic Regression", feature_columns, scaler, label_encoders)
            get_user_prediction(dt_model, "Decision Tree", feature_columns, scaler, label_encoders)
            
            again = input("\nPredict another? (y/n): ").strip().lower()
            if again != 'y':
                break
    else:
        print("\nTip: Use --predict or -p flag to enter your own data for prediction")
        print("Example: python main.py --predict")

if __name__ == "__main__":
    main()