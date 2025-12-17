import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import re
import numpy as np

def preprocess_data():
    df = pd.read_csv("data/data.csv")
    
    #defining X and y labels
    X = df.drop(['Disease'], axis=1)
    y = df['Disease']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    symptoms = X.columns.tolist()
    symptoms = [s.lower().strip() for s in X.columns]
    X.columns = symptoms
    
    # print("Available symptoms:", symptoms)
    
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

def normalize(text):   #normalizing text
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text    

def normalize_token(token):    #normalizing token
    if len(token) > 3 and token.endswith('s') and token not in {'loss', 'dizziness', 'breathless', 'weakness', 'tiredness'}:
        return token[:-1]
    return token

def text_to_symptom(user_text, symptoms):
    user_text = normalize(user_text)
    user_tokens = set(user_text.split())  #tokenizing user text
    user_tokens = {normalize_token(t) for t in user_tokens}  #normalizing user tokens
    user_tokens = expand_tokens(user_tokens)  #expanding user tokens

    stopwords = {'of', 'the', 'my', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'i', 'have'}
    user_tokens = user_tokens - stopwords
    
    vector = []
    matched_symptoms = []

    for symptom in symptoms:
        symptom_normalized = re.sub(r'[_\-]', ' ', symptom.lower())
        symptom_tokens = set(symptom_normalized.split())    #tokenizing symptoms dataset
        symptom_tokens = symptom_tokens - stopwords
        symptom_tokens = {normalize_token(t) for t in symptom_tokens}  #normalizing symptoms 
        symptom_tokens = expand_tokens(symptom_tokens)  #expanding symptoms

        # Calculate token overlap
        overlap = user_tokens.intersection(symptom_tokens)

        matched = 0
        
        # Matching logic based on symptom token count
        if len(symptom_tokens) == 1:
            # Single-word symptom: need exact match
            generic_words = {'pain', 'severe', 'mild', 'chronic', 'acute', 'high', 'low'}
            if overlap and list(overlap)[0] not in generic_words:
                matched = 1        
        
        elif len(symptom_tokens) == 2:
            # Two-word symptom: need BOTH words to match
            matched = 1 if len(overlap) == 2 else 0
        else:
            # Three+ word symptom: need at least 2 matching words
            required_matches = max(2, int(0.75 * len(symptom_tokens)))
            matched = 1 if len(overlap) >= required_matches else 0
        
        if matched:
            matched_symptoms.append(symptom)
        
        vector.append(matched)

    print(f"\nMatched {len(matched_symptoms)} symptoms: {matched_symptoms}")
    return np.array(vector).reshape(1, -1), len(matched_symptoms)

def expand_tokens(tokens):
    expanded = set()
    
    synonym_map = {        
        #body parts and systems
        "glucose": {"glucose", "sugar", "bloodsugar", "sugarlevel", "sugar level", "glucose level"},
        "bp": {"bp", "bloodpressure"},
        "abdominal": {"abdominal", "abdomen", "stomach", "belly", "tummy", "abdomin", "gastric"},
        "chest": {"chest", "thoracic"},
        "joint": {"joint", "joints", "elbow", "shoulder", "knee"},
        "head": {"head", "cranial", "skull"},
        "body": {"body", "torso", "trunk"},
        "back": {"back", "spine"},

        #symptoms
        "fever": {"fever", "feverish", "temperature", "pyrexia"},
        "fatigue": {"fatigue", "tired", "exhausted", "weak", "weakness", "tiredness", "lethargy"},
        "cough": {"cough", "coughing"},
        "headache": {"headache", "migraine"},
        "nausea": {"nausea", "nauseous", "queasy"},
        "vomit": {"vomit", "vomiting", "puking", "puke"},
        "diarrhea": {"diarrhea", "loose", "loose motion"},
        "rash": {"rash", "rashes"},
        "itch": {"itch", "itching", "itchy"},
        "swelling": {"swelling", "swollen", "edema"},
        "dizzy": {"dizzy", "dizziness", "vertigo", "lightheaded"},
        "backache": {"backache", "back"},
        
        "jaundice": {"jaundice", "yellow", "yellowing"},
        "dark": {"dark", "brown"},
        "clay": {"clay","pale"},
        "bloody": {"bloody", "blood"},
        "fatty": {"fatty", "oily", "greasy"},
        
        #severity/frequency
        "chronic": {"chronic", "persistent", "long", "term", "ongoing", "prolonged"},
        "excessive": {"excessive", "increased", "extreme", "always"},
        
        #specific symptoms
        "chill": {"chill", "rigor", "shaking", "violent"},
        "thirst": {"thirst", "thirsty", "polydipsia"},
        "vision": {"vision", "blurred", "blurry", "blur", "unclear"},
        "sweat": {"sweat", "sweating", "perspiration"},
        "heartburn": {"heartburn", "acid", "reflux"},
        "satiety": {"satiety", "full", "filling"},
        "epigastric": {"epigastric", "upper"},
        "defecation": {"defecation", "movement"},
        "intolerance": {"intolerance", "sensitive"},
        
        #respiratory
        "breath": {"breath", "breathing", "breathless"},
        "shortness": {"shortness", "short", "difficulty"},

        #weight
        "weight": {"weight", "mass"},
        "gain": {"gain", "gaining", "gained", "put"},
        
        #severity descriptors
        "slight": {"slight", "mild", "little", "minor"},
        "severe": {"severe", "intense", "extreme", "acute", "bad"},
        "constant": {"constant", "persistent", "ongoing", "continuous", "always"},

        "diarrhea": {"diarrhea","diarrhoea"},
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

    #---------------------------BASELINE MODELS-------------------------------
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_augmented, y_augmented)

    svm_baseline = LinearSVC(max_iter=10000)
    svm_baseline.fit(X_augmented, y_augmented)
    
    #--------------------------PCA + SVM ------------------------------------
    pca = PCA(n_components=0.95, random_state=42)

    X_train_pca = pca.fit_transform(X_augmented)  #feature reduction
    X_test_pca = pca.transform(X_test)

    svm_pca = LinearSVC(max_iter=10000)
    svm_pca.fit(X_train_pca, y_augmented)

    return X_train, X_test, y_train, y_test, nb_model, svm_baseline, svm_pca, pca