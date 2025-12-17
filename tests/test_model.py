import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score

def test_model_loading():
    model_dir = "models"
    
    assert os.path.exists(f"{model_dir}/naive_bayes_model.pkl")
    assert os.path.exists(f"{model_dir}/svm_baseline_model.pkl")
    assert os.path.exists(f"{model_dir}/svm_pca_model.pkl")
    
def test_model_prediction():
    model = joblib.load("models/naive_bayes_model.pkl")
    symptoms_list = joblib.load("models/symptoms_list.pkl")
    
    dummy_input = np.zeros((1, len(symptoms_list)))
    pred = model.predict(dummy_input)
    assert pred is not None
    assert len(pred) == 1
    
def test_model_accuracy():
    """Ensure models meet minimum accuracy threshold"""
    from app.ml_utils import preprocess_data, train_model
    
    X, y_encoded, le, symptoms = preprocess_data()
    _, X_test, _, y_test, nb_model, svm_baseline, svm_pca, pca = train_model(X, y_encoded)
    
    # Test accuracy thresholds
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    assert nb_acc > 0.85, f"NB accuracy {nb_acc} below threshold"