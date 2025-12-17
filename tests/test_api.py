from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Medical Assistant API Running..."

def test_predict_with_symptoms():
    response = client.post(
        "/predict",
        json={"symptoms_text": "I have fever cough and headache"}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_predict_insufficient_symptoms():
    response = client.post(
        "/predict",
        json={"symptoms_text": "fever"}
    )
    assert response.status_code == 200
    assert "warning" in response.json()