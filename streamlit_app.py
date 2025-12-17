import streamlit as st
import pandas as pd
import joblib
import os
from app.ml_utils import text_to_symptom

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Assistant AI", page_icon="⚕️")

# --- LOAD MODELS & UTILS ---
MODEL_DIR = os.path.join(os.getcwd(), 'models')

@st.cache_resource
def load_assets():
    nb = joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
    svm_baseline = joblib.load(os.path.join(MODEL_DIR, 'svm_baseline_model.pkl'))
    svm_pca = joblib.load(os.path.join(MODEL_DIR, 'svm_pca_model.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca_transform.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    symptoms_list = joblib.load(os.path.join(MODEL_DIR, 'symptoms_list.pkl'))
    return nb, svm_baseline, svm_pca, pca, le, symptoms_list

try:
    nb_model, svm_baseline, svm_pca, pca_transformer, le, symptoms_list = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- UI LAYOUT ---
st.title("⚕️ Medical Assistant AI")
st.markdown("Enter your symptoms below to get a prediction of potential conditions.")

# User Input
user_input = st.text_area("Describe how you are feeling:", 
                          placeholder="e.g., I have a high fever, a bad cough, and a headache...",
                          height=150)

if st.button("Submit Symptoms"):
    if not user_input.strip():
        st.warning("Please enter some symptoms first.")
    else:
        # Process input using your existing utility
        user_vector, matched_count = text_to_symptom(user_input, symptoms_list)
        
        if matched_count < 3:
            st.warning(f"Matched only {matched_count} symptoms. Please provide more detail (at least 3 symptoms required).")
        else:
            # Prepare data for prediction
            user_vector_df = pd.DataFrame(user_vector, columns=symptoms_list)
            
            # 1. Prediction (NB)
            nb_pred = nb_model.predict(user_vector_df)
            nb_result = le.inverse_transform(nb_pred)[0]
            
            # 2. Prediction (SVM PCA)
            user_vector_pca = pca_transformer.transform(user_vector_df)
            svm_pca_pred = svm_pca.predict(user_vector_pca)
            svm_pca_result = le.inverse_transform(svm_pca_pred)[0]
            
            # 3. Prediction (SVM Baseline)
            svm_base_pred = svm_baseline.predict(user_vector_df)
            svm_base_result = le.inverse_transform(svm_base_pred)[0]
            
            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader("Results")
            
            # Formatted according to your request: "You could have this, or this, or this"
            st.info(f"💡 **It is possible you could have:** {nb_result}")
            st.info(f"🔍 **Alternatively, it could be:** {svm_pca_result}")
            st.info(f"🩺 **Or perhaps:** {svm_base_result}")
            
            with st.expander("View matched symptoms"):
                st.write(f"Total matched: {matched_count}")