import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from app.auth import supabase
from pydantic import BaseModel
import joblib, os
from app.ml_utils import text_to_symptom

app = FastAPI(title="Medical Assistant AI")

#loading models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

nb_model = joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
svm_baseline_model = joblib.load(os.path.join(MODEL_DIR, 'svm_baseline_model.pkl'))
svm_pca_model = joblib.load(os.path.join(MODEL_DIR, 'svm_pca_model.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca_transform.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
symptoms_list = joblib.load(os.path.join(MODEL_DIR, 'symptoms_list.pkl'))

class symptomsInput(BaseModel):
    symptoms_text: str
    
@app.get("/")
def root():
    return {"message": "Medical Assistant API Running..."}
    
@app.post("/predict")
def predict_disease(data: symptomsInput):
    try:
        user_vector, matched_count = text_to_symptom(data.symptoms_text, symptoms_list)

        if matched_count < 3:
            return {
                "warning": "Need at least 3 symptoms for reliable prediction.",
                "matched_count": matched_count,
                "predictions": None
            }
            
        user_vector_df = pd.DataFrame(user_vector, columns=symptoms_list)

        nb_pred = nb_model.predict(user_vector_df)
        svm_pred = svm_baseline_model.predict(user_vector_df)
        user_vector_pca = pca.transform(user_vector_df)
        svm_pca_pred = svm_pca_model.predict(user_vector_pca)
        
        return {
            "matched_symptoms": matched_count,
            "predictions": {
                "naive_bayes": le.inverse_transform(nb_pred)[0],
                "linear_svm": le.inverse_transform(svm_pred)[0],
                "svm_pca": le.inverse_transform(svm_pca_pred)[0]
            }
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/signup")
def signup(email: str, password: str):
    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })
    if response.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")
    return {"message": "Signup Successful"}

@app.post("/login")
def login(email: str, password: str):
    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
    if response.user is None:
        raise HTTPException(status_code=400, detail="Invalid Credentials")
    return {
        "access_token": response.session.access_token,
        "token_type": "bearer"        
    }

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    content = await file.read()
    symptoms_text = content.decode("utf-8")

    matched, preds = predict_disease(symptoms_text)

    return {
        "input_type": "file",
        "matched_symptoms": matched,
        "predictions": preds
    }    
