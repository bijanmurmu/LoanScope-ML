# LoanScope-ML

LoanScope-ML is a Machine Learning project developed to predict whether a loan application will be approved or not based on applicant financial details such as income, credit history, and loan amount.

This project was developed as part of the **FSP**.

---

## Features
- Data preprocessing and cleaning
- Logistic Regression model
- Decision Tree classifier
- Accuracy comparison of models
- **Interactive prediction mode** - test with your own data
- **Custom dataset support** - use your own CSV files
- Simple and easy-to-understand code structure

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Project Structure
```
LoanScope-ML/
├── main.py
├── requirements.txt
├── README.md
├── dataset/
│   └── loan_approval.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train_logistic_regression.py
│   └── train_decision_tree.py
└── report/
```

---

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/LoanScope-ML.git
cd LoanScope-ML
```

### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 3: Run the Project
```bash
python main.py
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `--dataset`, `-d` | Path to CSV dataset (default: `dataset/loan_approval.csv`) |
| `--predict`, `-p` | Enable interactive prediction mode |
| `--help`, `-h` | Show help message |

### Examples

**Train models only:**
```bash
python main.py
```

**Train and predict with your data:**
```bash
python main.py --predict
```

**Use a custom dataset:**
```bash
python main.py --dataset path/to/your_data.csv --predict
```

---

## Dataset Format

Your CSV file should have the following structure:
- All feature columns first
- **Last column must be the target** (what you're predicting)
- Categorical and numeric columns are automatically handled

Example:
| name | city | income | credit_score | loan_amount | years_employed | points | loan_approved |
|------|------|--------|--------------|-------------|----------------|--------|---------------|
| John | NYC  | 50000  | 720          | 15000       | 5              | 45     | True          |

---

## Output

- Displays accuracy of Logistic Regression model
- Displays accuracy of Decision Tree model
- In prediction mode: Shows APPROVED ✓ or REJECTED ✗ for your input

Conclusion

Logistic Regression provides stable performance for loan approval prediction, while Decision Tree offers better interpretability. This project demonstrates how machine learning can assist financial institutions in decision-making.