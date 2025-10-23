import os
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, v1
from contextlib import asynccontextmanager
from glob import glob

# Training pipeline imports
from src.exception.exception import DeliveryTimeException
from src.components.data_ingestion import DataIngestion
from src.logging.logger import logging as training_logger  # Avoid conflict with main logger
from src.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import sys
import warnings

# Suppress Pydantic V2 warning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model artifacts and paths - defined at module level
model_artifacts = {}
PIPELINE_PATH = None
MODEL_PATH = None

def run_training_pipeline():
    """Execute the full ML training pipeline from main.py"""
    try:
        logger.info("Starting training pipeline...")
        trainingPipelineConfig = TrainingPipelineConfig()
        dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
        data_ingestion = DataIngestion(dataIngestionConfig)

        logger.info("Initiate the data ingestion")
        dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logger.info("DataIngestion Completed")
        print(dataIngestionArtifact)

        data_validation_config = DataValidationConfig(trainingPipelineConfig)
        data_validation = DataValidation(dataIngestionArtifact, data_validation_config)
        logger.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info("Data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingPipelineConfig)
        logger.info("Data Transformation Started")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logger.info("Data Transformation Completed")

        logger.info("Model Training Started")
        model_trainer_config = ModelTrainerConfig(trainingPipelineConfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logger.info("Model Training Artifact created")
        logger.info("Training pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise DeliveryTimeException(e, sys)

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

# Initialize paths - try config first, fall back to finding latest artifacts
logger.info("Initializing configuration and artifact paths...")
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
        else:
            PIPELINE_PATH = None
            logger.warning("No artifacts found via config or search")
            
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    logger.info("Attempting to find latest artifacts...")
    PIPELINE_PATH, _ = find_latest_artifacts()
    MODEL_PATH = "final_model/model.pkl"
    if not PIPELINE_PATH:
        logger.warning("No artifacts found")

logger.info(f"Initialized - PIPELINE_PATH: {PIPELINE_PATH}")
logger.info(f"Initialized - MODEL_PATH: {MODEL_PATH}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or train model artifacts on startup and cleanup on shutdown"""
    try:
        logger.info("Initializing application...")
        
        # Check if artifacts exist; if not, run training
        pipeline_exists = PIPELINE_PATH and os.path.exists(PIPELINE_PATH)
        model_exists = MODEL_PATH and os.path.exists(MODEL_PATH)
        
        if not pipeline_exists or not model_exists:
            logger.info("Artifacts not found. Running training pipeline...")
            run_training_pipeline()
            # After training, re-find artifacts (they should now exist)
            found_pipeline, _ = find_latest_artifacts()
            if found_pipeline:
                global PIPELINE_PATH
                PIPELINE_PATH = found_pipeline
                logger.info(f"Training generated artifacts. Using pipeline: {PIPELINE_PATH}")
            else:
                raise FileNotFoundError("Training completed but artifacts not found. Check logs.")
        else:
            logger.info("Artifacts found. Skipping training.")

        logger.info(f"Pipeline path: {PIPELINE_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")

        if not PIPELINE_PATH or not os.path.exists(PIPELINE_PATH):
            error_msg = f"Pipeline not found at {PIPELINE_PATH}\n"
            error_msg += "\nPlease ensure artifacts are available or training succeeds.\n"
            error_msg += "Training creates the pipeline at: Artifacts/<timestamp>/data_transformation/transformed_object/\n"
            
            # List what files exist in Artifacts if it exists
            artifacts_base = "Artifacts"
            if os.path.exists(artifacts_base):
                try:
                    artifact_dirs = os.listdir(artifacts_base)
                    error_msg += f"\nAvailable artifact directories: {artifact_dirs}\n"
                    if artifact_dirs:
                        latest_dir = sorted(artifact_dirs, reverse=True)[0]
                        pipeline_dir = os.path.join(artifacts_base, latest_dir, "data_transformation", "transformed_object")
                        if os.path.exists(pipeline_dir):
                            files = os.listdir(pipeline_dir)
                            error_msg += f"\nFiles in latest transformed_object directory: {files}\n"
                except Exception as list_error:
                    error_msg += f"\nError listing artifacts: {list_error}\n"
            
            raise FileNotFoundError(error_msg)
        
        if not MODEL_PATH or not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}\n"
                                   "Ensure model training saves to final_model/model.pkl")

        # Load the complete preprocessing pipeline
        logger.info(f"Loading pipeline from: {PIPELINE_PATH}")
        model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
        
        # Load the trained model
        logger.info(f"Loading model from: {MODEL_PATH}")
        model_artifacts['model'] = joblib.load(MODEL_PATH)

        logger.info("✓ Model artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model artifacts: {str(e)}")
        logger.error(f"Pipeline path used: {PIPELINE_PATH}")
        logger.error(f"Model path used: {MODEL_PATH}")
        raise

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

# Input schema with ORIGINAL feature names (using Pydantic V1 compatibility for schema_extra)
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
        # Use v1 compatibility for schema_extra
        json_schema_extra = {
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
        message="ETA Prediction API is running" if model_loaded else "Model not loaded",
        model_loaded=model_loaded
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_eta(input_data: DeliveryInput):
    try:
        if 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline or model not loaded"
            )

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
    try:
        if 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
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
    if 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": "1.0.0",
        "model_type": type(model_artifacts['model']).__name__,
        "pipeline_type": type(model_artifacts['pipeline']).__name__
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
