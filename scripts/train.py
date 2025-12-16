import pandas as pd
from app.ml_utils import preprocess_data, train_model
import joblib

def main():
    X, y_encoded, le, symptoms = preprocess_data()
    X_train, X_test, y_train, y_test, nb_model, svm_baseline, svm_pca, pca = train_model(X, y_encoded)    

    #saving models
    joblib.dump(nb_model, '../models/naive_bayes_model.pkl')
    joblib.dump(svm_baseline, '../models/svm_baseline_model.pkl')
    joblib.dump(svm_pca, '../models/svm_pca_model.pkl')
    joblib.dump(pca, '../models/pca_transform.pkl')
    joblib.dump(le, '../models/label_encoder.pkl')
    joblib.dump(symptoms, '../models/symptoms_list.pkl')

if __name__ == '__main__': 
    main()