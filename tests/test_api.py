from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_text():
    response = client.post(
        "/predict/text",
        json={"symptoms_text": "I have fever and headache"}
    )
    assert response.status_code == 200
