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
    X_train, X_test, y_train, y_test, nb_model, lsvm_model = train_model(X, y_encoded)

    user_text = input("Enter your symptoms: ").lower()
    user_vector = text_to_symptom(user_text, symptoms)

    #making predictions
    model_evaluation(nb_model, lsvm_model, y_test, X_test, X_train, y_train)
    
    real_time_prediction(user_vector, X, nb_model, lsvm_model, le)

def preprocess_data():
    df = pd.read_csv("data.csv")
    
    #defining X and y labels
    X = df.drop(['Disease'], axis=1)
    y = df['Disease']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # print(df.columns)
    
    symptoms = X.columns.tolist()
    symptoms = [s.lower().strip() for s in X.columns]
    
    print(symptoms)
    
    return X,y_encoded, le, symptoms

def augment_symptoms(X, y, noise_level=0.1):
    """
    Randomly flip some symptom bits to simulate real user input.
    noise_level: fraction of symptoms to flip per row
    """
    X_aug = X.copy()
    for i in range(len(X_aug)):
        # how many symptoms to flip for this row
        n_flip = max(1, int(noise_level * X.shape[1]))
        flip_indices = np.random.choice(X.shape[1], n_flip, replace=False)
        # flip 0 -> 1 or 1 -> 0
        X_aug.iloc[i, flip_indices] = 1 - X_aug.iloc[i, flip_indices]
    
    return X_aug, y


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
    
    X_augmented, y_augmented = augment_symptoms(X_train, y_train, noise_level=0.1)

    nb_model = MultinomialNB()
    nb_model.fit(X_augmented, y_augmented)

    lsvm_model = LinearSVC(max_iter=10000)
    lsvm_model.fit(X_augmented, y_augmented)

    return X_train, X_test, y_train, y_test, nb_model, lsvm_model
    
def model_evaluation(nb_model, lsvm_model, y_test, X_test, X_train, y_train):
    y_pred_nb = nb_model.predict(X_test)    
    y_pred_svm = lsvm_model.predict(X_test)
    
    print("TEST ACCURACY:\n")
    print(f"NB Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    
    y_train_pred_nb = nb_model.predict(X_train)
    y_train_pred_svm = lsvm_model.predict(X_train)
    
    print("TRAIN ACCURACY:\n")
    train_acc_nb = accuracy_score(y_train, y_train_pred_nb)
    train_acc_svm = accuracy_score(y_train, y_train_pred_svm)
    
    print(f"NB Training Accuracy: {train_acc_nb:.4f}")
    print(f"SVM Training Accuracy: {train_acc_svm:.4f}")
    
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
    
def real_time_prediction(user_vector, X, nb_model, lsvm_model, le):
    user_vector = pd.DataFrame(user_vector, columns=X.columns)

    nb_pred = nb_model.predict(user_vector)
    svm_pred = lsvm_model.predict(user_vector)
    
    print(f"\nNB SAYS: {le.inverse_transform(nb_pred)[0]}")
    print(f"\nSVM SAYS: {le.inverse_transform(svm_pred)[0]}")


if __name__ == '__main__': 
    main()

