import os
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from src.entity.config_entity import DataTransformationConfig, TrainingPipelineConfig
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logging.logger import logging
from glob import glob
from src.exception.exception import DeliveryTimeException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model artifacts and training status
model_artifacts = {}
is_model_trained = False

# Auto-training flag - set to False if you want to skip auto-training
AUTO_TRAIN_ON_STARTUP = True  # Set to False to avoid startup delays

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

def run_training_pipeline():
    """Execute the complete training pipeline from main.py with detailed error handling"""
    error_details = {}
    
    try:
        logger.info("=== Starting Training Pipeline ===")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python executable: {sys.executable}")
        
        # Step 0: Validate config initialization
        try:
            logger.info("Step 0: Initializing TrainingPipelineConfig...")
            trainingPipelineConfig = TrainingPipelineConfig()
            logger.info(f"✓ TrainingPipelineConfig created: {type(trainingPipelineConfig)}")
            error_details['config_init'] = 'success'
        except Exception as e:
            error_details['config_init'] = f"Failed: {str(e)}"
            logger.error(f"Config initialization failed: {str(e)}", exc_info=True)
            raise DeliveryTimeException(f"TrainingPipelineConfig initialization failed: {str(e)}", sys)
        
        # Step 1: Data Ingestion
        try:
            logger.info("Step 1: Initializing DataIngestionConfig...")
            dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
            logger.info(f"✓ DataIngestionConfig created")
            
            logger.info("Step 1: Creating DataIngestion instance...")
            data_ingestion = DataIngestion(dataIngestionConfig)
            logger.info(f"✓ DataIngestion instance created: {type(data_ingestion)}")
            
            logger.info("Step 1: Calling initiate_data_ingestion()...")
            dataIngestionArtifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"✓ Data Ingestion completed successfully")
            logger.info(f"DataIngestionArtifact type: {type(dataIngestionArtifact)}")
            error_details['data_ingestion'] = 'success'
            print(f"DataIngestionArtifact: {dataIngestionArtifact}")
            
        except Exception as e:
            error_details['data_ingestion'] = f"Failed: {str(e)}"
            logger.error(f"Data Ingestion failed: {str(e)}", exc_info=True)
            raise DeliveryTimeException(f"Data ingestion failed: {str(e)}", sys)
        
        # Step 2: Data Validation
        try:
            logger.info("Step 2: Initializing DataValidationConfig...")
            data_validation_config = DataValidationConfig(trainingPipelineConfig)
            logger.info(f"✓ DataValidationConfig created")
            
            logger.info("Step 2: Creating DataValidation instance...")
            data_validation = DataValidation(dataIngestionArtifact, data_validation_config)
            logger.info(f"✓ DataValidation instance created: {type(data_validation)}")
            
            logger.info("Step 2: Calling initiate_data_validation()...")
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(f"✓ Data Validation completed successfully")
            logger.info(f"DataValidationArtifact type: {type(data_validation_artifact)}")
            error_details['data_validation'] = 'success'
            print(f"DataValidationArtifact: {data_validation_artifact}")
            
        except Exception as e:
            error_details['data_validation'] = f"Failed: {str(e)}"
            logger.error(f"Data Validation failed: {str(e)}", exc_info=True)
            raise DeliveryTimeException(f"Data validation failed: {str(e)}", sys)
        
        # Step 3: Data Transformation
        try:
            logger.info("Step 3: Initializing DataTransformationConfig...")
            data_transformation_config = DataTransformationConfig(trainingPipelineConfig)
            logger.info(f"✓ DataTransformationConfig created")
            
            logger.info("Step 3: Creating DataTransformation instance...")
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logger.info(f"✓ DataTransformation instance created: {type(data_transformation)}")
            
            logger.info("Step 3: Calling initiate_data_transformation()...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"✓ Data Transformation completed successfully")
            logger.info(f"DataTransformationArtifact type: {type(data_transformation_artifact)}")
            error_details['data_transformation'] = 'success'
            print(f"DataTransformationArtifact: {data_transformation_artifact}")
            
        except Exception as e:
            error_details['data_transformation'] = f"Failed: {str(e)}"
            logger.error(f"Data Transformation failed: {str(e)}", exc_info=True)
            raise DeliveryTimeException(f"Data transformation failed: {str(e)}", sys)
        
        # Step 4: Model Training
        try:
            logger.info("Step 4: Initializing ModelTrainerConfig...")
            model_trainer_config = ModelTrainerConfig(trainingPipelineConfig)
            logger.info(f"✓ ModelTrainerConfig created")
            
            logger.info("Step 4: Creating ModelTrainer instance...")
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config, 
                data_transformation_artifact=data_transformation_artifact
            )
            logger.info(f"✓ ModelTrainer instance created: {type(model_trainer)}")
            
            logger.info("Step 4: Calling initiate_model_trainer()...")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"✓ Model Training completed successfully")
            logger.info(f"ModelTrainerArtifact type: {type(model_trainer_artifact)}")
            error_details['model_trainer'] = 'success'
            print(f"ModelTrainerArtifact: {model_trainer_artifact}")
            
        except Exception as e:
            error_details['model_trainer'] = f"Failed: {str(e)}"
            logger.error(f"Model Training failed: {str(e)}", exc_info=True)
            raise DeliveryTimeException(f"Model training failed: {str(e)}", sys)
        
        logger.info("=== Training Pipeline Completed Successfully ===")
        logger.info(f"Pipeline execution summary: {error_details}")
        return True, error_details
        
    except Exception as e:
        logger.error(f"=== Training Pipeline FAILED ===")
        logger.error(f"Pipeline execution summary: {error_details}")
        logger.error(f"Final error: {str(e)}", exc_info=True)
        return False, error_details

def load_model_artifacts():
    """Load model artifacts after training or on startup"""
    global PIPELINE_PATH, model_artifacts, is_model_trained
    
    try:
        # Try config path first
        try:
            training_pipeline_config = TrainingPipelineConfig()
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            PIPELINE_PATH = data_transformation_config.transformed_object_file_path
            MODEL_PATH = "final_model/model.pkl"
            logger.info(f"Using config paths - Pipeline: {PIPELINE_PATH}, Model: {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Config path failed: {e}, falling back to latest artifacts")
            # Fallback to finding latest artifacts
            PIPELINE_PATH, _ = find_latest_artifacts()
            MODEL_PATH = "final_model/model.pkl"
        
        # Check if artifacts exist
        pipeline_exists = PIPELINE_PATH and os.path.exists(PIPELINE_PATH)
        model_exists = os.path.exists(MODEL_PATH)
        
        logger.info(f"Artifact check - Pipeline exists: {pipeline_exists}, Model exists: {model_exists}")
        logger.info(f"Pipeline path: {PIPELINE_PATH}")
        logger.info(f"Model path: {MODEL_PATH}")
        
        if not pipeline_exists or not model_exists:
            logger.warning("Model artifacts not found.")
            if not pipeline_exists:
                logger.warning(f"Pipeline file missing: {PIPELINE_PATH}")
            if not model_exists:
                logger.warning(f"Model file missing: {MODEL_PATH}")
            return False
        
        # Load the complete preprocessing pipeline
        logger.info(f"Loading pipeline from: {PIPELINE_PATH}")
        model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
        logger.info(f"✓ Pipeline loaded: {type(model_artifacts['pipeline'])}")
        
        # Load the trained model
        logger.info(f"Loading model from: {MODEL_PATH}")
        model_artifacts['model'] = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model loaded: {type(model_artifacts['model'])}")
        
        is_model_trained = True
        logger.info("✓ Model artifacts loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}", exc_info=True)
        logger.error(f"Pipeline path attempted: {PIPELINE_PATH}")
        logger.error(f"Model path attempted: {MODEL_PATH}")
        is_model_trained = False
        return False

# Auto-training on startup
def initialize_model():
    """Initialize model by checking for existing or training new"""
    global is_model_trained
    
    logger.info("=== Model Initialization Started ===")
    
    # First, try to load existing model
    if load_model_artifacts():
        logger.info("✓ Existing model loaded on startup")
        logger.info("=== Model Initialization Completed Successfully ===")
        return True
    
    # If no model exists and auto-training is enabled, train new model
    if AUTO_TRAIN_ON_STARTUP:
        logger.info("No existing model found. Starting automatic training...")
        start_time = datetime.now()
        
        success, error_details = run_training_pipeline()
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        if success:
            logger.info(f"✓ Automatic training completed in {training_duration:.2f} seconds")
            
            # Load the newly trained model
            if load_model_artifacts():
                logger.info("✓ Newly trained model loaded successfully")
                logger.info("=== Model Initialization Completed Successfully ===")
                return True
            else:
                logger.error("Training succeeded but model loading failed")
                logger.error(f"Error details: {error_details}")
                logger.info("=== Model Initialization FAILED ===")
                return False
        else:
            logger.error(f"Automatic training failed after {training_duration:.2f} seconds")
            logger.error(f"Error details: {error_details}")
            logger.info("=== Model Initialization FAILED ===")
            return False
    else:
        logger.warning("Auto-training disabled. Model not available until /train is called")
        logger.info("=== Model Initialization Skipped (Auto-training disabled) ===")
        return False

# Initialize model on startup
if __name__ == "__main__":
    # Run initialization before starting FastAPI
    initialize_model()
elif "PYTEST_CURRENT_TEST" not in os.environ:
    # For non-test environments, initialize on import
    initialize_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan - model is already loaded during initialization"""
    logger.info("FastAPI application starting...")
    
    # Model is already loaded/initialized at this point
    if is_model_trained:
        logger.info(f"✓ Model is ready for inference")
        logger.info(f"Pipeline type: {type(model_artifacts.get('pipeline', 'None'))}")
        logger.info(f"Model type: {type(model_artifacts.get('model', 'None'))}")
    else:
        logger.warning("⚠️  Model not available - API will require manual training")
    
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
    artifacts_path: str
    pipeline_summary: Dict[str, str]

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
    auto_trained: bool = None

# Training endpoint (for manual retraining) with detailed error reporting
@app.post("/train", response_model=TrainingResponse, tags=["Training"])
def train_model(request: TrainingRequest):
    """Manual model retraining endpoint with detailed error tracking"""
    global model_artifacts, is_model_trained, PIPELINE_PATH
    
    start_time = datetime.now()
    error_details = {}
    
    try:
        logger.info("=== Manual Training Requested ===")
        
        if is_model_trained and not request.force_retrain:
            # Find the latest artifacts to provide path info
            current_pipeline, _ = find_latest_artifacts()
            current_model_path = MODEL_PATH
            
            artifacts_path = os.path.dirname(current_pipeline) if current_pipeline else current_model_path
            
            return TrainingResponse(
                status="skipped",
                message="Model already trained. Use force_retrain=True to retrain.",
                model_version="1.0.0",
                training_duration=0.0,
                artifacts_path=artifacts_path,
                pipeline_summary={"status": "skipped", "existing_model": True}
            )
        
        logger.info("Clearing existing model state for retraining...")
        model_artifacts.clear()
        is_model_trained = False
        
        # Run training pipeline with detailed tracking
        logger.info("Executing training pipeline...")
        success, pipeline_summary = run_training_pipeline()
        training_time = (datetime.now() - start_time).total_seconds()
        
        if not success:
            logger.error(f"Training pipeline failed. Summary: {pipeline_summary}")
            raise HTTPException(
                status_code=500,
                detail=f"Training pipeline failed. Check logs for details. Pipeline summary: {pipeline_summary}"
            )
        
        logger.info("Training pipeline completed successfully. Loading model artifacts...")
        
        # Load the newly trained model
        load_start_time = datetime.now()
        if load_model_artifacts():
            load_duration = (datetime.now() - load_start_time).total_seconds()
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Get artifacts path
            try:
                training_pipeline_config = TrainingPipelineConfig()
                data_transformation_config = DataTransformationConfig(training_pipeline_config)
                artifacts_base = os.path.dirname(data_transformation_config.transformed_object_file_path)
                artifacts_path = os.path.dirname(artifacts_base)
            except Exception as e:
                logger.warning(f"Could not determine artifacts path from config: {e}")
                artifacts_path = os.path.dirname(PIPELINE_PATH) if PIPELINE_PATH else "unknown"
            
            logger.info(f"Model loading completed in {load_duration:.2f} seconds")
            logger.info(f"Total training + loading time: {total_duration:.2f} seconds")
            
            return TrainingResponse(
                status="success",
                message="Model trained and loaded successfully",
                model_version="1.0.0",
                training_duration=total_duration,
                artifacts_path=artifacts_path,
                pipeline_summary=pipeline_summary
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail="Training succeeded but model loading failed. Check if artifacts were created correctly."
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"=== Manual Training FAILED ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}", exc_info=True)
        logger.error(f"Pipeline summary: {error_details}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Training failed with error: {type(e).__name__}: {str(e)}. Check server logs for full traceback."
        )

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
        model_loaded=model_loaded,
        training_required=not is_model_trained,
        auto_trained=AUTO_TRAIN_ON_STARTUP and is_model_trained
    )

@app.get("/diagnostics", tags=["Debug"])
def get_diagnostics():
    """Get diagnostic information about the current state"""
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        config_pipeline_path = data_transformation_config.transformed_object_file_path
        
        latest_pipeline, _ = find_latest_artifacts()
        
        return {
            "is_model_trained": is_model_trained,
            "model_artifacts_loaded": bool(model_artifacts),
            "pipeline_path_config": config_pipeline_path,
            "pipeline_path_latest": latest_pipeline,
            "model_path": "final_model/model.pkl",
            "config_exists": os.path.exists(config_pipeline_path),
            "latest_artifacts_exist": bool(latest_pipeline),
            "model_file_exists": os.path.exists("final_model/model.pkl"),
            "artifacts_dir_exists": os.path.exists("Artifacts"),
            "current_working_dir": os.getcwd(),
            "python_path": sys.executable,
            "auto_train_enabled": AUTO_TRAIN_ON_STARTUP
        }
    except Exception as e:
        return {
            "error": str(e),
            "is_model_trained": is_model_trained,
            "model_artifacts_loaded": bool(model_artifacts),
            "config_load_failed": True
        }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_eta(input_data: DeliveryInput):
    global model_artifacts, is_model_trained
    
    if not is_model_trained or 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please call /train endpoint first."
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
        raise HTTPException(status_code=503, detail="Model not available. Please call /train endpoint first.")
    
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
        raise HTTPException(status_code=503, detail="Model not available. Please call /train endpoint first.")
    
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        config_path = data_transformation_config.transformed_object_file_path
    except:
        config_path = "Config path unavailable"
    
    return {
        "model_version": "1.0.0",
        "model_type": type(model_artifacts['model']).__name__,
        "pipeline_type": type(model_artifacts['pipeline']).__name__,
        "pipeline_path": PIPELINE_PATH or config_path,
        "model_path": "final_model/model.pkl",
        "auto_trained": AUTO_TRAIN_ON_STARTUP
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    uvicorn.run(app, host="0.0.0.0", port=8000)
