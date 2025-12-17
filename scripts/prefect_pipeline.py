import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from datetime import datetime
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
from app.ml_utils import preprocess_data, train_model, augment_symptoms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from scripts.notifications import send_discord_notification

# ======================== TASK 1: DATA INGESTION ========================
@task(
    name="ingest_data",
    description="Load and validate raw medical symptom data",
    retries=3,
    retry_delay_seconds=5,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def ingest_data(file_path: str = "data/data.csv"):
    """
    Load CSV data with validation checks
    """
    try:
        print(f"📥 Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Validation checks
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"
        assert "Disease" in df.columns, "Missing target column 'Disease'"
        assert len(df) > 100, f"Dataset too small: {len(df)} samples"
        assert df['Disease'].nunique() >= 5, f"Too few diseases: {df['Disease'].nunique()}"
        
        print(f"✅ Data loaded: {len(df)} samples, {df['Disease'].nunique()} diseases")
        return df
    
    except Exception as e:
        print(f"❌ Data ingestion failed: {str(e)}")
        raise


# ======================== TASK 2: FEATURE ENGINEERING ========================
@task(
    name="engineer_features",
    description="Extract features and prepare data for training",
    retries=2,
    retry_delay_seconds=3
)
def engineer_features(df: pd.DataFrame):
    """
    Preprocess data: encode labels, extract symptoms
    """
    try:
        print("🔧 Engineering features...")
        
        X, y_encoded, le, symptoms = preprocess_data()
        
        print(f"✅ Features prepared:")
        print(f"   - Samples: {len(X)}")
        print(f"   - Symptoms (features): {len(symptoms)}")
        print(f"   - Disease classes: {len(le.classes_)}")
        
        return X, y_encoded, le, symptoms
    
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        raise


# ======================== TASK 3: MODEL TRAINING ========================
@task(
    name="train_models",
    description="Train Naive Bayes, Linear SVM, and SVM+PCA models",
    retries=2,
    retry_delay_seconds=10
)
def train_models(X, y_encoded):
    """
    Train all three models with data augmentation
    """
    try:
        print("🤖 Training models...")
        
        X_train, X_test, y_train, y_test, nb_model, svm_baseline, svm_pca, pca = train_model(X, y_encoded)
        
        print("✅ Models trained successfully:")
        print(f"   - Naive Bayes")
        print(f"   - Linear SVM (Baseline)")
        print(f"   - Linear SVM + PCA")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "nb_model": nb_model,
            "svm_baseline": svm_baseline,
            "svm_pca": svm_pca,
            "pca": pca
        }
    
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        raise


# ======================== TASK 4: MODEL EVALUATION ========================
@task(
    name="evaluate_models",
    description="Evaluate model performance on test set",
    retries=1
)
def evaluate_models(models_dict):
    """
    Calculate accuracy and generate evaluation metrics
    """
    try:
        print("📊 Evaluating models...")
        
        nb_model = models_dict["nb_model"]
        svm_baseline = models_dict["svm_baseline"]
        svm_pca = models_dict["svm_pca"]
        pca = models_dict["pca"]
        X_train = models_dict["X_train"]
        X_test = models_dict["X_test"]
        y_train = models_dict["y_train"]
        y_test = models_dict["y_test"]
        
        # Predictions
        y_pred_nb = nb_model.predict(X_test)
        y_pred_svm = svm_baseline.predict(X_test)
        
        X_test_pca = pca.transform(X_test)
        X_train_pca = pca.transform(X_train)
        y_pred_svm_pca = svm_pca.predict(X_test_pca)
        
        # Calculate metrics
        metrics = {
            "naive_bayes": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_nb)),
                "train_accuracy": float(accuracy_score(y_train, nb_model.predict(X_train)))
            },
            "svm_baseline": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_svm)),
                "train_accuracy": float(accuracy_score(y_train, svm_baseline.predict(X_train)))
            },
            "svm_pca": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_svm_pca)),
                "train_accuracy": float(accuracy_score(y_train, svm_pca.predict(X_train_pca)))
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n✅ EVALUATION RESULTS:")
        print(f"   Naive Bayes Test Accuracy: {metrics['naive_bayes']['test_accuracy']:.4f}")
        print(f"   SVM Baseline Test Accuracy: {metrics['svm_baseline']['test_accuracy']:.4f}")
        print(f"   SVM + PCA Test Accuracy: {metrics['svm_pca']['test_accuracy']:.4f}")
        
        # Quality checks
        assert metrics['naive_bayes']['test_accuracy'] > 0.70, "NB accuracy below threshold"
        assert metrics['svm_baseline']['test_accuracy'] > 0.70, "SVM accuracy below threshold"
        
        return metrics
    
    except Exception as e:
        print(f"❌ Model evaluation failed: {str(e)}")
        raise


# ======================== TASK 5: SAVE & VERSION MODELS ========================
@task(
    name="save_models",
    description="Save trained models with versioning",
    retries=2,
    retry_delay_seconds=3
)
def save_models(models_dict, le, symptoms, metrics):
    """
    Save models with timestamp-based versioning
    """
    try:
        print("💾 Saving models...")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Create versioned directory
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(MODEL_DIR, f"version_{version}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Save models to version directory
        joblib.dump(models_dict["nb_model"], os.path.join(version_dir, 'naive_bayes_model.pkl'))
        joblib.dump(models_dict["svm_baseline"], os.path.join(version_dir, 'svm_baseline_model.pkl'))
        joblib.dump(models_dict["svm_pca"], os.path.join(version_dir, 'svm_pca_model.pkl'))
        joblib.dump(models_dict["pca"], os.path.join(version_dir, 'pca_transform.pkl'))
        joblib.dump(le, os.path.join(version_dir, 'label_encoder.pkl'))
        joblib.dump(symptoms, os.path.join(version_dir, 'symptoms_list.pkl'))
        
        # Save metrics
        with open(os.path.join(version_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Also save to main models directory (for API to use)
        joblib.dump(models_dict["nb_model"], os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
        joblib.dump(models_dict["svm_baseline"], os.path.join(MODEL_DIR, 'svm_baseline_model.pkl'))
        joblib.dump(models_dict["svm_pca"], os.path.join(MODEL_DIR, 'svm_pca_model.pkl'))
        joblib.dump(models_dict["pca"], os.path.join(MODEL_DIR, 'pca_transform.pkl'))
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        joblib.dump(symptoms, os.path.join(MODEL_DIR, 'symptoms_list.pkl'))
        
        print(f"✅ Models saved to:")
        print(f"   - Production: {MODEL_DIR}")
        print(f"   - Versioned: {version_dir}")
        
        return version_dir
    
    except Exception as e:
        print(f"❌ Model saving failed: {str(e)}")
        raise


# ======================== NOTIFICATION TASK ========================
@task(
    name="send_notification",
    description="Send Discord notification about pipeline status",
    retries=3,
    retry_delay_seconds=5
)
def send_notification(success: bool, metrics: dict = None, error_msg: str = None):
    """
    Send Discord notification about pipeline status
    Falls back to console output if webhook is not configured
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    
    if success:
        print("\n" + "="*60)
        print("🎉 ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        if metrics:
            print(f"📊 Final Metrics:")
            print(f"   NB Accuracy: {metrics['naive_bayes']['test_accuracy']:.4f}")
            print(f"   SVM Accuracy: {metrics['svm_baseline']['test_accuracy']:.4f}")
            print(f"   SVM+PCA Accuracy: {metrics['svm_pca']['test_accuracy']:.4f}")
        print("="*60 + "\n")
        
        # Send Discord notification if webhook is configured
        if webhook_url:
            send_discord_notification(webhook_url, success=True, metrics=metrics)
        else:
            print("ℹ️  Discord webhook not configured, skipping Discord notification")
    else:
        print("\n" + "="*60)
        print("❌ ML PIPELINE FAILED!")
        print("="*60)
        if error_msg:
            print(f"Error: {error_msg}")
        print("="*60 + "\n")
        
        # Send Discord notification if webhook is configured
        if webhook_url:
            send_discord_notification(webhook_url, success=False, error_msg=error_msg)
        else:
            print("ℹ️  Discord webhook not configured, skipping Discord notification")


# ======================== MAIN PREFECT FLOW ========================
@flow(
    name="Medical ML Training Pipeline",
    description="End-to-end ML workflow with Prefect orchestration",
    log_prints=True
)
def ml_training_pipeline(data_path: str = "data/data.csv"):
    """
    Complete ML pipeline orchestrated by Prefect
    """
    try:
        print("\n🚀 Starting ML Training Pipeline...")
        print("="*60)
        
        # Step 1: Data Ingestion
        df = ingest_data(data_path)
        
        # Step 2: Feature Engineering
        X, y_encoded, le, symptoms = engineer_features(df)
        
        # Step 3: Model Training
        models_dict = train_models(X, y_encoded)
        
        # Step 4: Model Evaluation
        metrics = evaluate_models(models_dict)
        
        # Step 5: Save & Version Models
        version_dir = save_models(models_dict, le, symptoms, metrics)
        
        # Step 6: Success Notification
        send_notification(success=True, metrics=metrics)
        
        return {
            "status": "success",
            "metrics": metrics,
            "version_dir": version_dir
        }
    
    except Exception as e:
        # Failure Notification
        send_notification(success=False, error_msg=str(e))
        raise


if __name__ == "__main__":
    # Run the pipeline
    result = ml_training_pipeline()
    print(f"\n✅ Pipeline Result: {result['status']}")