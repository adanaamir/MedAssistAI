import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from auth import supabase
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from ml_utils import text_to_symptom, symptoms_list, le

app = FastAPI(title="Medical Assistant AI")

#loading models
nb_model = joblib.load('models/naive_bayes_model.pkl')
svm_pca_model = joblib.load('models/svm_pca_model.pkl')
pca = joblib.load('models/pca_transform.pkl')

@app.get("/")
def root():
    return {"message": "Medical Assistant API Running..."}
    
# @app.route("/predict")
# def predict_disease(data: symptoms_list):
        
    
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
    
def main():
    X, y_encoded, le, symptoms = preprocess_data()
    X_train, X_test, y_train, y_test, nb_model, svm_baseline, svm_pca, pca = train_model(X, y_encoded)

    save_model(nb_model, svm_pca, pca)
    
    while True:
        user_text = input("\nEnter your symptoms: ").lower()
        user_vector, matched_count = text_to_symptom(user_text, symptoms)

        if matched_count < 3:
            print("\nWarning! Need at least 3 symptoms for reliable prediction.")
            print("Please enter more symptoms.\n")
        else:
            break
    
    model_evaluation(nb_model, svm_baseline, svm_pca, pca, y_test, X_test, X_train, y_train)
    
    real_time_prediction(user_vector, X, nb_model, svm_baseline, svm_pca, pca, le)
    
    
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
    
def real_time_prediction(user_vector, X, nb_model, svm_baseline, svm_pca, pca, le):
    user_vector_df = pd.DataFrame(user_vector, columns=X.columns)

    nb_pred = nb_model.predict(user_vector_df)
    svm_pred = svm_baseline.predict(user_vector_df)
    
    user_vector_pca = pca.transform(user_vector_df)
    svm_pca_pred = svm_pca.predict(user_vector_pca)
    
    print(f"\nNB SAYS: {le.inverse_transform(nb_pred)[0]}")
    print(f"SVM SAYS: {le.inverse_transform(svm_pred)[0]}")
    print(f"SVM (PCA) SAYS: {le.inverse_transform(svm_pca_pred)[0]}")
    
def save_model(nb_model, svm_pca, pca):
    joblib.dump(nb_model, 'naive_bayes_model.pkl')
    joblib.dump(svm_pca, 'svm_pca_model.pkl')
    joblib.dump(pca, 'pca_transform.pkl')
    
    print("Models and PCA saved successfully!")

if __name__ == '__main__': 
    main()