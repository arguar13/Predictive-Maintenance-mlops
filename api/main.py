import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from api.schemas import BookingFeatures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hotel Market Segmentation API",
    description="MLOps API for multiclass market segment classification",
    version="1.0.0"
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    """
    Loads the latest production model from MLflow Model Registry upon server startup.
    """
    global model
    try:
        model_name = "HotelSegmentClassifier"
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully into the API.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")

@app.post("/predict")
def predict_segment(features: BookingFeatures):
    """
    Predicts the market segment based on booking features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    try:
        # Convert Pydantic model to DataFrame for the scikit-learn pipeline
        input_data = pd.DataFrame([features.dict()])
        
        # Perform inference
        prediction = model.predict(input_data)
        
        return {
            "predicted_market_segment": str(prediction[0])
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))