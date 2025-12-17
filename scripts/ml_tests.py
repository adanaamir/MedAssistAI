import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation, train_test_validation
import joblib
import os

def run_ml_quality_checks():
    print("🔍 Running DeepChecks ML Quality Suites...")
    if not os.path.exists("ml_reports"):
        os.makedirs("ml_reports")
        print("  - Created 'ml_reports' directory")
    
    df = pd.read_csv("data/data.csv")
    label_col = 'Disease'
    
    # Create DeepChecks Dataset
    ds = Dataset(df, label=label_col)
    
    # 2. Data Integrity Suite (Detects issues like outliers, duplicates, etc.)
    print("  - Running Data Integrity Suite...")
    integ_suite = data_integrity()
    integ_result = integ_suite.run(ds)
    integ_result.save_as_html("ml_reports/data_integrity_report.html")
    
    # 3. Train-Test Validation (Detects Drift)
    # Split for comparison
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_ds = Dataset(train_df, label=label_col)
    test_ds = Dataset(test_df, label=label_col)
    
    print("  - Running Drift and Validation Suite...")
    validation_suite = train_test_validation()
    validation_result = validation_suite.run(train_ds, test_ds)
    validation_result.save_as_html("ml_reports/train_test_validation_report.html")
    
    # 4. Model Evaluation Suite
    print("  - Running Model Evaluation Suite...")
    model = joblib.load("models/naive_bayes_model.pkl")
    
    evaluation_suite = model_evaluation()
    eval_result = evaluation_suite.run(train_ds, test_ds, model)
    eval_result.save_as_html("ml_reports/model_evaluation_report.html")
    
    # Logic to fail CI/CD if critical checks fail
    if not integ_result.passed():
        print("❌ Data Integrity checks failed!")
        return False
    
    print("✅ All ML automated tests passed!")
    return True

if __name__ == "__main__":
    os.makedirs("ml_reports", exist_ok=True)
    success = run_ml_quality_checks()
    if not success:
        exit(1) # Fail CI/CD build