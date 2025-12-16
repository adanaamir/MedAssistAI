import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from app.auth import supabase
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib, os
from ml_utils import text_to_symptom, symptoms_list, le

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
    
@app.route("/predict")
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
                "linear_svm": le.inverse_transform(svm_baseline_model)[0],
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
def signup(email: str, password: str):
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

    
def model_evaluation(nb_model, svm_baseline, svm_pca, pca, y_test, X_test, X_train, y_train):
    #BASELINE PREDICTIONS
    y_pred_nb = nb_model.predict(X_test)    
    y_pred_svm = svm_baseline.predict(X_test)
    
    #PCA PREDICTIONS
    X_test_pca = pca.transform(X_test)
    X_train_pca = pca.transform(X_train)
    y_pred_svm_pca = svm_pca.predict(X_test_pca)
    
    print("\nTEST ACCURACY")
    print(f"NB Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"SVM Baseline Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print(f"SVM + PCA Accuracy: {accuracy_score(y_test, y_pred_svm_pca):.4f}")
    
    print("TRAIN ACCURACY:\n")
    print(f"NB Training Accuracy: {accuracy_score(y_train, nb_model.predict(X_train)):.4f}")
    print(f"SVM Training Accuracy: {accuracy_score(y_train, svm_baseline.predict(X_train)):.4f}")
    print(f"SVM + PCA Accuracy: {accuracy_score(y_train, svm_pca.predict(X_train_pca)):.4f}")

    print("\n===============FINAL PREDICTIONS================")
    print("\nMultinomial Naives Baye's Model")
    print(f"accuracy Score: {accuracy_score(y_test, y_pred_nb)}")
    # print(f"Classification Report:\n {classification_report(y_test, y_pred_nb)}")
    # print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_nb)}")

    print("Linear SVM Model")
    print(f"accuracy Score: {accuracy_score(y_test, y_pred_svm)}")
    
    print("Linear SVM (PCA) Model")
    print(f"accuracy Score: {accuracy_score(y_test, y_pred_svm_pca)}")
    # print(f"Classification Report:\n {classification_report(y_test, y_pred_svm)}")
    # print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_svm)}")
