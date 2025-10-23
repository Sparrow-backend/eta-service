import os
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from src.entity.config_entity import DataTransformationConfig, TrainingPipelineConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model artifacts and training status
model_artifacts = {}
is_model_trained = False

def find_latest_artifacts():
    """Find the most recent artifacts directory"""
    try:
        artifacts_base = "Artifacts"
        if not os.path.exists(artifacts_base):
            return None, None
        
        # Get all artifact directories sorted by timestamp (most recent first)
        artifact_dirs = sorted(
            [d for d in os.listdir(artifacts_base) if os.path.isdir(os.path.join(artifacts_base, d))],
            reverse=True
        )
        
        if not artifact_dirs:
            return None, None
        
        latest_dir = artifact_dirs[0]
        logger.info(f"Found latest artifacts directory: {latest_dir}")
        
        # Look for the pipeline file (try common names)
        pipeline_dir = os.path.join(artifacts_base, latest_dir, "data_transformation", "transformed_object")
        
        if os.path.exists(pipeline_dir):
            # Try to find the pipeline file with various possible names
            possible_names = ["preprocessing.pkl", "preprocessor.pkl", "pipeline.pkl", "transformed_object.pkl"]
            
            for name in possible_names:
                pipeline_path = os.path.join(pipeline_dir, name)
                if os.path.exists(pipeline_path):
                    logger.info(f"Found pipeline at: {pipeline_path}")
                    return pipeline_path, None
            
            # If no standard name found, get any .pkl file
            pkl_files = glob(os.path.join(pipeline_dir, "*.pkl"))
            if pkl_files:
                logger.info(f"Found pipeline file: {pkl_files[0]}")
                return pkl_files[0], None
        
        return None, None
        
    except Exception as e:
        logger.error(f"Error finding artifacts: {e}")
        return None, None

# Try to use config first, fall back to finding latest artifacts
try:
    training_pipeline_config = TrainingPipelineConfig()
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    PIPELINE_PATH = data_transformation_config.transformed_object_file_path
    MODEL_PATH = "final_model/model.pkl"
    
    # Check if the config path exists
    if not os.path.exists(PIPELINE_PATH):
        logger.warning(f"Config path not found: {PIPELINE_PATH}")
        logger.info("Attempting to find latest artifacts...")
        found_pipeline, _ = find_latest_artifacts()
        if found_pipeline:
            PIPELINE_PATH = found_pipeline
            logger.info(f"Using found pipeline: {PIPELINE_PATH}")
        
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    logger.info("Attempting to find latest artifacts...")
    PIPELINE_PATH, _ = find_latest_artifacts()
    MODEL_PATH = "final_model/model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup and cleanup on shutdown"""
    global is_model_trained
    
    try:
        logger.info("Initializing application...")
        
        # Check if model exists and load it
        if PIPELINE_PATH and os.path.exists(PIPELINE_PATH) and os.path.exists(MODEL_PATH):
            logger.info("Loading existing model artifacts...")
            logger.info(f"Pipeline path: {PIPELINE_PATH}")
            logger.info(f"Model path: {MODEL_PATH}")

            # Load the complete preprocessing pipeline
            model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
            
            # Load the trained model
            model_artifacts['model'] = joblib.load(MODEL_PATH)

            is_model_trained = True
            logger.info("✓ Model artifacts loaded successfully")
        else:
            logger.warning("No trained model found. Training will be required.")
            is_model_trained = False
            
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        is_model_trained = False

    yield

    # Cleanup
    logger.info("Shutting down application...")
    model_artifacts.clear()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ETA Prediction API",
    version="1.0.0",
    description="API for predicting delivery time based on various features",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Training request model
class TrainingRequest(BaseModel):
    force_retrain: bool = Field(default=False, description="Force retraining even if model exists")

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_version: str
    training_duration: float

# Input schema with ORIGINAL feature names
class DeliveryInput(BaseModel):
    Distance_km: float = Field(..., gt=0, example=10.5, description="Distance in kilometers")
    Courier_Experience_yrs: float = Field(..., ge=0, example=2.0, description="Courier experience in years")
    Vehicle_Type: str = Field(..., example="Scooter", description="Vehicle type")
    Weather: str = Field(..., example="Sunny", description="Weather condition")
    Time_of_Day: str = Field(..., example="Morning", description="Time of day")
    Traffic_Level: str = Field(..., example="Low", description="Traffic level")

    @field_validator('Distance_km')
    def validate_distance(cls, v):
        if v > 1000:
            raise ValueError('Distance cannot exceed 1000 km')
        return v
    
    @field_validator('Courier_Experience_yrs')
    def validate_experience(cls, v):
        if v > 50:
            raise ValueError('Experience cannot exceed 50 years')
        return v
    
    @field_validator('Vehicle_Type')
    def validate_vehicle(cls, v):
        allowed = ['Scooter', 'Pickup Truck', 'Motorcycle']
        if v not in allowed:
            raise ValueError(f'Vehicle type must be one of {allowed}')
        return v
    
    @field_validator('Weather')
    def validate_weather(cls, v):
        allowed = ['Sunny', 'Rainy', 'Foggy', 'Snowy', 'Windy']
        if v not in allowed:
            raise ValueError(f'Weather must be one of {allowed}')
        return v
    
    @field_validator('Time_of_Day')
    def validate_time(cls, v):
        allowed = ['Morning', 'Afternoon', 'Evening', 'Night']
        if v not in allowed:
            raise ValueError(f'Time of day must be one of {allowed}')
        return v
    
    @field_validator('Traffic_Level')
    def validate_traffic(cls, v):
        allowed = ['Low', 'Medium', 'High']
        if v not in allowed:
            raise ValueError(f'Traffic level must be one of {allowed}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "Distance_km": 10.5,
                "Courier_Experience_yrs": 2.0,
                "Vehicle_Type": "Scooter",
                "Weather": "Sunny",
                "Time_of_Day": "Morning",
                "Traffic_Level": "Low"
            }
        }

class PredictionResponse(BaseModel):
    predicted_delivery_time: float
    input_features: Dict[str, Any]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    training_required: bool

# Training endpoint
@app.post("/train", response_model=TrainingResponse, tags=["Training"])
def train_model(request: TrainingRequest):
    """Trigger model training and load the model for inference"""
    global model_artifacts, is_model_trained
    
    if is_model_trained and not request.force_retrain:
        return TrainingResponse(
            status="skipped",
            message="Model already trained. Use force_retrain=True to retrain.",
            model_version="1.0.0",
            training_duration=0.0
        )
    
    start_time = pd.Timestamp.now()
    
    try:
        logger.info("Starting model training...")
        
        # Initialize training pipeline
        training_pipeline_config = TrainingPipelineConfig()
        
        # Execute complete training pipeline
        logger.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion(training_pipeline_config)
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        logger.info("Step 2: Data Validation")
        data_validation = DataValidation(train_data_path, training_pipeline_config)
        status = data_validation.initiate_data_validation()
        
        logger.info("Step 3: Data Transformation")
        data_transformation = DataTransformation(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            training_pipeline_config=training_pipeline_config
        )
        train_arr, test_arr, preprocessor_obj = data_transformation.initiate_data_transformation()
        
        logger.info("Step 4: Model Training")
        model_trainer = ModelTrainer(
            model_data=train_arr,
            preprocess_object=preprocessor_obj,
            training_pipeline_config=training_pipeline_config
        )
        trained_model = model_trainer.initiate_model_trainer()
        
        logger.info("Step 5: Model Evaluation")
        model_evaluation = ModelEvaluation(
            model=trained_model,
            X_train=train_arr[:, :-1],
            X_test=test_arr[:, :-1],
            y_train=train_arr[:, -1],
            y_test=test_arr[:, -1],
            training_pipeline_config=training_pipeline_config
        )
        model_score, model_accuracy_score = model_evaluation.initiate_model_evaluation()
        
        logger.info("Step 6: Model Pushing")
        model_pusher = ModelPusher(
            model=trained_model,
            preprocess_object=preprocessor_obj,
            model_evaluation=model_evaluation,
            training_pipeline_config=training_pipeline_config
        )
        model_pusher.initiate_model_pusher()
        
        # Load the newly trained model into memory
        end_time = pd.Timestamp.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Update paths to new artifacts
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        PIPELINE_PATH = data_transformation_config.transformed_object_file_path
        MODEL_PATH = "final_model/model.pkl"
        
        model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
        model_artifacts['model'] = joblib.load(MODEL_PATH)
        is_model_trained = True
        
        logger.info(f"Model training completed successfully in {training_duration:.2f} seconds")
        
        return TrainingResponse(
            status="success",
            message="Model trained and loaded successfully",
            model_version="1.0.0",
            training_duration=training_duration
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    model_loaded = (
        'pipeline' in model_artifacts 
        and model_artifacts['pipeline'] is not None
        and 'model' in model_artifacts
        and model_artifacts['model'] is not None
    )
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        message="ETA Prediction API is running" if model_loaded else "Model not loaded - training required",
        model_loaded=model_loaded,
        training_required=not is_model_trained
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_eta(input_data: DeliveryInput):
    global model_artifacts, is_model_trained
    
    if not is_model_trained or 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not trained. Please call /train endpoint first."
        )

    try:
        input_dict = input_data.model_dump()
        logger.info(f"Prediction request: {input_dict}")

        # Create DataFrame and transform
        input_df = pd.DataFrame([input_dict])
        transformed_input = model_artifacts['pipeline'].transform(input_df)
        
        # Predict
        prediction = model_artifacts['model'].predict(transformed_input)
        predicted_time = float(prediction[0])

        logger.info(f"Prediction: {predicted_time:.2f} minutes")

        return PredictionResponse(
            predicted_delivery_time=round(predicted_time, 2),
            input_features=input_dict,
            model_version="1.0.0"
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(input_data: list[DeliveryInput]):
    global model_artifacts, is_model_trained
    
    if not is_model_trained or 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not trained. Please call /train endpoint first.")
    
    try:
        predictions = []
        for idx, data in enumerate(input_data):
            input_dict = data.model_dump()
            input_df = pd.DataFrame([input_dict])
            transformed = model_artifacts['pipeline'].transform(input_df)
            pred = model_artifacts['model'].predict(transformed)
            
            predictions.append({
                "index": idx,
                "predicted_delivery_time": round(float(pred[0]), 2),
                "input_features": input_dict
            })
        
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["Model"])
def get_model_info():
    global model_artifacts, is_model_trained
    
    if not is_model_trained or 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not trained. Please call /train endpoint first.")
    
    return {
        "model_version": "1.0.0",
        "model_type": type(model_artifacts['model']).__name__,
        "pipeline_type": type(model_artifacts['pipeline']).__name__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
