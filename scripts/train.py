import sys,  os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib 
import pandas as pd
from app.ml_utils import preprocess_data, train_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def model_evaluation(nb_model, svm_baseline, svm_pca, pca, y_test, X_test, X_train, y_train):
    #BASELINE PREDICTIONS
    y_pred_nb = nb_model.predict(X_test)    
    y_pred_svm = svm_baseline.predict(X_test)
    
    # PCA PREDICTIONS
    X_test_pca = pca.transform(X_test)
    X_train_pca = pca.transform(X_train)
    y_pred_svm_pca = svm_pca.predict(X_test_pca)
    
    print("\nTEST ACCURACY")
    print(f"NB Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"SVM Baseline Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print(f"SVM + PCA Accuracy: {accuracy_score(y_test, y_pred_svm_pca):.4f}")
    
    print("\nTRAIN ACCURACY")
    print(f"NB Training Accuracy: {accuracy_score(y_train, nb_model.predict(X_train)):.4f}")
    print(f"SVM Training Accuracy: {accuracy_score(y_train, svm_baseline.predict(X_train)):.4f}")
    print(f"SVM + PCA Accuracy: {accuracy_score(y_train, svm_pca.predict(X_train_pca)):.4f}")

    print("\nFINAL PREDICTIONS SUMMARY")
    print(f"\nMultinomial Naive Bayes Model")
    print(f"  Accuracy Score: {accuracy_score(y_test, y_pred_nb):.4f}")
    #print(f"Classification Report:\n {classification_report(y_test, y_pred_nb)}")
    # print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_nb)}")
    
    print(f"\nLinear SVM Model")
    print(f"  Accuracy Score: {accuracy_score(y_test, y_pred_svm):.4f}")
    #print(f"Classification Report:\n {classification_report(y_test, y_pred_svm)}")
    #print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred_svm)}")

    print(f"\nLinear SVM (PCA) Model")
    print(f"  Accuracy Score: {accuracy_score(y_test, y_pred_svm_pca):.4f}")

def main():
    print("Starting model training...")
    
    df = pd.read_csv("data/data.csv")
    original_symptoms = df.drop(['Disease'], axis=1).columns.tolist()
    
    #preprocess data
    X, y_encoded, le, symptoms = preprocess_data()
    print(f"Dataset loaded: {len(X)} samples, {len(symptoms)} symptoms")
    
    #train models
    X_train, X_test, y_train, y_test, nb_model, svm_baseline, svm_pca, pca = train_model(X, y_encoded)
    print("Models trained successfully!")
    
    #evaluate models
    model_evaluation(nb_model, svm_baseline, svm_pca, pca, y_test, X_test, X_train, y_train)
    
    #define model directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
    
    #create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    #save models
    print("\nSaving models...")
    joblib.dump(nb_model, os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
    joblib.dump(svm_baseline, os.path.join(MODEL_DIR, 'svm_baseline_model.pkl'))
    joblib.dump(svm_pca, os.path.join(MODEL_DIR, 'svm_pca_model.pkl'))
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca_transform.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(original_symptoms, os.path.join(MODEL_DIR, 'symptoms_list.pkl'))
    
    print("All models saved successfully!")
    print(f"Models saved to: {os.path.abspath(MODEL_DIR)}")

if __name__ == '__main__': 
    main()