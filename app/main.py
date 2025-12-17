import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from app.auth import supabase
from pydantic import BaseModel
import joblib, os, logging
from app.ml_utils import text_to_symptom

app = FastAPI(title="Medical Assistant AI")

#logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

#loading models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

try:
    logger.info("Loading ML models...")
    nb_model = joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
    svm_baseline_model = joblib.load(os.path.join(MODEL_DIR, 'svm_baseline_model.pkl'))
    svm_pca_model = joblib.load(os.path.join(MODEL_DIR, 'svm_pca_model.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca_transform.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    symptoms_list = joblib.load(os.path.join(MODEL_DIR, 'symptoms_list.pkl'))
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

class symptomsInput(BaseModel):
    symptoms_text: str
    
@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Medical Assistant API Running..."}
    
@app.post("/predict")
def predict_disease(data: symptomsInput):
    logger.info(f"Prediction request received with symptoms: {data.symptoms_text[:50]}")
    try:
        user_vector, matched_count = text_to_symptom(data.symptoms_text, symptoms_list)
        logger.info(f"Matched {matched_count} symptoms from input")
        
        if matched_count < 3:
            logger.warning(f"Insufficient symptoms matched: {matched_count} < 3")
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
        
        predictions = {
            "naive_bayes": le.inverse_transform(nb_pred)[0],
            "linear_svm": le.inverse_transform(svm_pred)[0],
            "svm_pca": le.inverse_transform(svm_pca_pred)[0]
        }
        
        logger.info(f"Predictions generated: {predictions}")
        
        return {
            "matched_symptoms": matched_count,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    logger.info(f"File upload received: {file.filename}")
    try:
        content = await file.read()
        symptoms_text = content.decode("utf-8")
        logger.info(f"File decoded successfully, length: {len(symptoms_text)} characters")

        data = symptomsInput(symptoms_text=symptoms_text)
        result = predict_disease(data)
        
        result["input_type"] = "file"
        logger.info("File prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"File prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/signup")
def signup(email: str, password: str):
    logger.info(f"Signup attempt for email: {email}")
    
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if response.user is None:
            logger.warning(f"Signup failed for: {email}")
            raise HTTPException(status_code=400, detail="Signup failed")
        
        logger.info(f"Signup successful for: {email}")
        return {"message": "Signup Successful"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Signup error")

@app.post("/login")
def login(email: str, password: str):
    logger.info(f"Login attempt for email: {email}")
    
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response.user is None:
            logger.warning(f"Login failed - invalid credentials for: {email}")
            raise HTTPException(status_code=400, detail="Invalid Credentials")
        
        logger.info(f"Login successful for: {email}")
        return {
            "access_token": response.session.access_token,
            "token_type": "bearer"        
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login error")
    
    
#YAYYYYYYYYYYYYYYYYY
#ok