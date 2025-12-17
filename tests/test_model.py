import joblib
import numpy as np
import os

def test_model_loading():
    model_dir = "models"
    
    assert os.path.exists(f"{model_dir}/naive_bayes_model.pkl")
    assert os.path.exists(f"{model_dir}/svm_baseline_model.pkl")
    assert os.path.exists(f"{model_dir}/svm_pca_model.pkl")
    
def test_model_prediction():
    model = joblib.load("models/naive_bayes_model.pkl")
    symptoms_list = joblib.load("models/symptoms_list.pkl")
    
    # Use correct shape based on actual symptom count
    dummy_input = np.zeros((1, len(symptoms_list)))
    pred = model.predict(dummy_input)
    assert pred is not None
    assert len(pred) == 1