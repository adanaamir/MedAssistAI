import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from scripts.prefect_pipeline import (
    ingest_data,
    engineer_features,
    train_models,
    evaluate_models,
    save_models
)
from scripts.notifications import send_discord_notification


def test_data_ingestion():
    """Test that data ingestion task works"""
    df = ingest_data("data/data.csv")
    
    assert df is not None
    assert len(df) > 0
    assert "Disease" in df.columns
    assert df.isnull().sum().sum() == 0


def test_feature_engineering():
    """Test feature engineering task"""
    df = ingest_data("data/data.csv")
    X, y_encoded, le, symptoms = engineer_features(df)
    
    assert X is not None
    assert len(symptoms) > 0
    assert len(X) == len(y_encoded)
    assert len(le.classes_) >= 5


def test_model_training():
    """Test that all three models train successfully"""
    df = ingest_data("data/data.csv")
    X, y_encoded, le, symptoms = engineer_features(df)
    models_dict = train_models(X, y_encoded)
    
    assert "nb_model" in models_dict
    assert "svm_baseline" in models_dict
    assert "svm_pca" in models_dict
    assert "pca" in models_dict
    assert models_dict["nb_model"] is not None


def test_model_evaluation():
    """Test that evaluation produces valid metrics"""
    df = ingest_data("data/data.csv")
    X, y_encoded, le, symptoms = engineer_features(df)
    models_dict = train_models(X, y_encoded)
    metrics = evaluate_models(models_dict)
    
    assert "naive_bayes" in metrics
    assert "svm_baseline" in metrics
    assert "svm_pca" in metrics
    
    # Check accuracy is between 0 and 1
    assert 0 <= metrics["naive_bayes"]["test_accuracy"] <= 1
    assert 0 <= metrics["svm_baseline"]["test_accuracy"] <= 1
    assert 0 <= metrics["svm_pca"]["test_accuracy"] <= 1
    
    # Check minimum accuracy threshold
    assert metrics["naive_bayes"]["test_accuracy"] > 0.70
    assert metrics["svm_baseline"]["test_accuracy"] > 0.70


def test_model_saving():
    """Test that models are saved correctly"""
    df = ingest_data("data/data.csv")
    X, y_encoded, le, symptoms = engineer_features(df)
    models_dict = train_models(X, y_encoded)
    metrics = evaluate_models(models_dict)
    version_dir = save_models(models_dict, le, symptoms, metrics)
    
    # Check version directory was created
    assert os.path.exists(version_dir)
    assert "version_" in version_dir
    
    # Check all models were saved
    assert os.path.exists(os.path.join(version_dir, "naive_bayes_model.pkl"))
    assert os.path.exists(os.path.join(version_dir, "svm_baseline_model.pkl"))
    assert os.path.exists(os.path.join(version_dir, "metrics.json"))


def test_discord_notification_success():
    """Test that Discord success notifications work without webhook"""
    # This should not raise any errors even without webhook URL
    send_discord_notification(
        webhook_url="",  # Empty URL for testing
        success=True,
        metrics={
            'naive_bayes': {'test_accuracy': 0.95},
            'svm_baseline': {'test_accuracy': 0.93},
            'svm_pca': {'test_accuracy': 0.94}
        }
    )


def test_discord_notification_failure():
    """Test that Discord failure notifications work without webhook"""
    # This should not raise any errors even without webhook URL
    send_discord_notification(
        webhook_url="",
        success=False,
        error_msg="Test error message for pipeline failure"
    )


def test_discord_notification_with_env():
    """Test Discord notification when webhook is in environment"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    
    # Should handle gracefully whether webhook exists or not
    send_discord_notification(
        webhook_url=webhook_url,
        success=True,
        metrics={
            'naive_bayes': {'test_accuracy': 0.92},
            'svm_baseline': {'test_accuracy': 0.90},
            'svm_pca': {'test_accuracy': 0.91}
        }
    )


def test_pipeline_error_handling():
    """Test that pipeline handles invalid data gracefully"""
    with pytest.raises(Exception):
        # Try to ingest non-existent file
        ingest_data("data/nonexistent.csv")


def test_metrics_threshold():
    """Test that models meet minimum performance thresholds"""
    df = ingest_data("data/data.csv")
    X, y_encoded, le, symptoms = engineer_features(df)
    models_dict = train_models(X, y_encoded)
    metrics = evaluate_models(models_dict)
    
    # All models should achieve at least 70% accuracy
    assert metrics["naive_bayes"]["test_accuracy"] >= 0.70, \
        f"NB accuracy {metrics['naive_bayes']['test_accuracy']} below 0.70"
    assert metrics["svm_baseline"]["test_accuracy"] >= 0.70, \
        f"SVM accuracy {metrics['svm_baseline']['test_accuracy']} below 0.70"
    assert metrics["svm_pca"]["test_accuracy"] >= 0.70, \
        f"SVM+PCA accuracy {metrics['svm_pca']['test_accuracy']} below 0.70"