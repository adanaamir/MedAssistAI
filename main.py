import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from auth import supabase
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

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

    user_text = input("\nEnter your symptoms: ").lower()
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
    
    symptoms = X.columns.tolist()
    symptoms = [s.lower().strip() for s in X.columns]
    
    print("Available symptoms:", symptoms)
    
    return X, y_encoded, le, symptoms

def augment_symptoms(X, y, core_frac=0.7, optional_frac=0.3, noise_level=0.01):
    """
    Creates a diverse and medically realistic dataset from binary symptom data.
    
    X: original symptom DataFrame
    y: disease labels
    core_frac: fraction of core symptoms to keep for each disease
    optional_frac: fraction of other symptoms to randomly add
    noise_level: small fraction of random flips to simulate user input mistakes
    """
    X_aug = pd.DataFrame(columns=X.columns)
    y_aug = []

    for idx, row in X.iterrows():
        disease = y[idx]
        symptoms_present = row[row == 1].index.tolist()
        symptoms_absent = row[row == 0].index.tolist()

        # Core symptoms: keep most of them
        n_core = max(1, int(core_frac * len(symptoms_present)))
        core_keep = np.random.choice(symptoms_present, n_core, replace=False).tolist()

        # Optional symptoms: randomly add some absent ones
        n_optional = max(0, int(optional_frac * len(symptoms_absent)))
        optional_add = np.random.choice(symptoms_absent, n_optional, replace=False).tolist()

        new_row = [0] * len(X.columns)
        for i, col in enumerate(X.columns):
            if col in core_keep or col in optional_add:
                new_row[i] = 1

        # Add small random noise
        n_noise = max(1, int(noise_level * len(X.columns)))
        noise_indices = np.random.choice(len(X.columns), n_noise, replace=False)
        for i in noise_indices:
            if new_row[i] == 0:
                new_row[i] = 1

        X_aug.loc[len(X_aug)] = new_row
        y_aug.append(disease)

    return X_aug, np.array(y_aug)

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text    

def normalize_token(token):
    if len(token) > 3 and token.endswith('s') and token not in {'loss', 'dizziness', 'breathless', 'weakness', 'tiredness'}:
        return token[:-1]
    return token

def text_to_symptom(user_text, symptoms):
    user_text = normalize(user_text)
    user_tokens = set(user_text.split())  #tokenizing user text
    user_tokens = {normalize_token(t) for t in user_tokens}  #normalizing user tokens
    user_tokens = expand_tokens(user_tokens)  #expanding user tokens

    vector = []
    matched_symptoms = []

    for symptom in symptoms:
        symptom_normalized = re.sub(r'[_\-]', ' ', symptom.lower())
        symptom_tokens = set(symptom_normalized.split())    #tokenizing symptoms dataset
        symptom_tokens = {normalize_token(t) for t in symptom_tokens}  #normalizing symptoms 
        symptom_tokens = expand_tokens(symptom_tokens)  #expanding symptoms

        # Calculate token overlap
        overlap = user_tokens.intersection(symptom_tokens)

        # Matching logic based on symptom token count
        if len(symptom_tokens) == 1:
            # Single-word symptom: need exact match
            matched = 1 if len(overlap) >= 1 else 0
        elif len(symptom_tokens) == 2:
            # Two-word symptom: need BOTH words to match
            matched = 1 if len(overlap) == 2 else 0
        else:
            # Three+ word symptom: need at least 2 matching words
            matched = 1 if len(overlap) >= 2 else 0
        
        if matched:
            matched_symptoms.append(symptom)
        
        vector.append(matched)

    print(f"\n✓ Matched {len(matched_symptoms)} symptoms: {matched_symptoms}")
    return np.array(vector).reshape(1, -1)

def expand_tokens(tokens):
    expanded = set()
    
    synonym_map = {
        #modifiers
        "high": {"high", "elevated", "raised", "increased", "very"},
        "low": {"low", "decreased", "reduced", "dropped"},
        
        #body parts and systems
        "glucose": {"glucose", "sugar", "bloodsugar", "sugarlevel"},
        "bp": {"bp", "bloodpressure", "blood", "pressure"},
        "abdominal": {"abdominal", "abdomen", "stomach", "belly", "tummy", "abdomin"},
        "chest": {"chest", "thoracic"},
        "joint": {"joint", "joints"},
        "head": {"head", "cranial"},
        "body": {"body", "torso", "trunk"},

        #symptoms
        "fever": {"fever", "feverish", "temperature", "pyrexia"},
        "fatigue": {"fatigue", "tired", "exhausted", "weak", "weakness", "tiredness", "lethargy"},
        "pain": {"pain", "ache", "aching", "hurt", "hurting", "painful", "discomfort"},
        "cough": {"cough", "coughing"},
        "headache": {"headache", "migraine"},
        "nausea": {"nausea", "nauseous", "queasy"},
        "vomit": {"vomit", "vomiting", "puking", "puke"},
        "diarrhea": {"diarrhea", "loose"},
        "rash": {"rash", "rashes", "eruption"},
        "itch": {"itch", "itching", "itchy"},
        "swelling": {"swelling", "swollen", "edema"},
        "dizzy": {"dizzy", "dizziness", "vertigo", "lightheaded"},
        "backache": {"backache", "back", "hurts"},
        
        #respiratory
        "breath": {"breath", "breathing", "breathless"},
        "shortness": {"shortness", "short", "difficulty"},
        
        #weight
        "weight": {"weight", "mass"},
        "loss": {"loss", "losing", "lost"},
        "gain": {"gain", "gaining", "gained"},
        
        #severity descriptors
        "slight": {"slight", "mild", "little", "minor"},
        "severe": {"severe", "intense", "extreme", "acute", "bad"},
        "constant": {"constant", "persistent", "ongoing", "continuous", "always"},
    }

    for token in tokens:
        normalized = normalize_token(token)
        found = False
        for base_term, variants in synonym_map.items():
            if normalized in variants or token in variants:
                expanded.add(base_term)
                found = True
                break
        
        if not found:
            expanded.add(token)

    return expanded
        
def train_model(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, train_size=0.8, random_state=42, stratify=y_encoded)
    
    y_train = pd.Series(y_train).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    X_augmented, y_augmented = augment_symptoms(X_train, y_train, noise_level=0.01)

    nb_model = MultinomialNB(alpha=0.1)
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
    
def real_time_prediction(user_vector, X, nb_model, lsvm_model, le):
    user_vector = pd.DataFrame(user_vector, columns=X.columns)

    nb_pred = nb_model.predict(user_vector)
    svm_pred = lsvm_model.predict(user_vector)
    
    print(f"\nNB SAYS: {le.inverse_transform(nb_pred)[0]}")
    print(f"\nSVM SAYS: {le.inverse_transform(svm_pred)[0]}")

if __name__ == '__main__': 
    main()