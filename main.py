import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from auth import supabase
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Medical Assistant API Running..."}
    
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
    y_test, nb_model, lsvm_model = train_model(X, y_encoded)

    user_text = input("Enter your symptoms: ").lower()
    user_vector = text_to_symptom(user_text, symptoms)

    #making predictions
    model_evaluation(nb_model, lsvm_model, y_test, user_vector)

def preprocess_data():
    df = pd.read_csv("data.csv")
    
    #defining X and y labels
    X = df.drop(['Disease'], axis=1)
    y = df['Disease']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # print(dict(zip(le.classes_, le.transform(le.classes_))))
    # print(df.columns)
    
    symptoms = X.columns.tolist()
    
    return X,y_encoded, le, symptoms

def text_to_symptom(user_text,symptoms):
    vector = []
    
    for symptom in symptoms:
        if symptom in user_text:
            vector.append(1)
        else:
            vector.append(0)
            
    print(vector)
        
    return np.array(vector).reshape(1, -1)
    
def train_model(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, train_size=0.8, random_state=42, stratify=y_encoded)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    lsvm_model = LinearSVC()
    lsvm_model.fit(X_train, y_train)
    
    #transforming back to disease names
    # y_pred_labels = le.inverse_transform(y_pred_labels)

    return y_test, nb_model, lsvm_model
    
def model_evaluation(nb_model, lsvm_model, y_test, user_vector):
    y_pred_nb = nb_model.predict(user_vector)
    print(y_pred_nb)
    
    y_pred_svm = lsvm_model.predict(user_vector)
    print(y_pred_svm)
    
    print("Multinomial Naives Baye's Model")
    print(f"accuracy Score: {accuracy_score(y_test, y_pred_nb)}")
    # print(f"Classification Report:\n {classification_report(y_test, y_pred_nb)}")
    # print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_nb)}")
    
    print("---------------------------------------------------------------------")

    print("Linear SVM Model")
    print(f"accuracy Score: {accuracy_score(y_test, y_pred_svm)}")
    # print(f"Classification Report:\n {classification_report(y_test, y_pred_svm)}")
    # print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_svm)}")
    
    print("BOTH TRAINED")


if __name__ == '__main__': 
    main()

