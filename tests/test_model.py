import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("models/naive_bayes_model.pkl")
    dummy_input = np.zeros((1, 50))
    pred = model.predict(dummy_input)
    assert pred is not None
