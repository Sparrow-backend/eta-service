import os
import joblib
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import tempfile
import shutil
from datetime import datetime
import zipfile

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
training_lock = False  # Prevent concurrent training

def find_latest_artifacts():
    """Find the most recent artifacts directory"""
    try:
        artifacts_base = "Artifacts"
        if not os.path.exists(artifacts_base):
            return None, None
        artifact_dirs = sorted(
            [d for d in os.listdir(artifacts_base) if os.path.isdir(os.path.join(artifacts_base, d))],
            reverse=True
        )
        if not artifact_dirs:
            return None, None
        latest_dir = artifact_dirs[0]
        logger.info(f"Found latest artifacts directory: {latest_dir}")
        pipeline_dir = os.path.join(artifacts_base, latest_dir, "data_transformation", "transformed_object")
        if os.path.exists(pipeline_dir):
            possible_names = ["preprocessing.pkl", "preprocessor.pkl", "pipeline.pkl", "transformed_object.pkl"]
            for name in possible_names:
                pipeline_path = os.path.join(pipeline_dir, name)
                if os.path.exists(pipeline_path):
                    logger.info(f"Found pipeline at: {pipeline_path}")
                    return pipeline_path, None
            pkl_files = glob(os.path.join(pipeline_dir, "*.pkl"))
            if pkl_files:
                logger.info(f"Found pipeline file: {pkl_files[0]}")
                return pkl_files[0], None
        return None, None
    except Exception as e:
        logger.error(f"Error finding artifacts: {e}")
        return None, None

# Try to initialize config and find existing artifacts
try:
    training_pipeline_config = TrainingPipelineConfig()
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    PIPELINE_PATH = data_transformation_config.transformed_object_file_path
    MODEL_PATH = "final_model/model.pkl"
    if not os.path.exists(PIPELINE_PATH):
        logger.warning(f"Config path not found: {PIPELINE_PATH}")
        found_pipeline, _ = find_latest_artifacts()
        if found_pipeline:
            PIPELINE_PATH = found_pipeline
            logger.info(f"Using found pipeline: {PIPELINE_PATH}")
except Exception as e:
    logger.warning(f"Could not load config: {e}")
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
            
            model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
            model_artifacts['model'] = joblib.load(MODEL_PATH)
            is_model_trained = True
            logger.info("✓ Existing model artifacts loaded successfully")
        else:
            logger.warning("No trained model found. Training will be required.")
            is_model_trained = False
            
    except Exception as e:
        logger.error(f"Failed to load existing model artifacts: {str(e)}")
        is_model_trained = False

    yield

    # Cleanup
    logger.info("Shutting down application...")
    model_artifacts.clear()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ETA Prediction API",
    version="1.0.0",
    description="API for training and predicting delivery time based on various features",
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

# Training request models
class TrainingRequest(BaseModel):
    """Request model for triggering training"""
    force_retrain: bool = Field(default=False, description="Force retraining even if model exists")

class TrainingConfig(BaseModel):
    """Configuration for training process"""
    test_size: float = Field(0.2, ge=0.1, le=0.4, description="Test split size")
    random_state: int = Field(42, description="Random state for reproducibility")
    max_train_samples: Optional[int] = Field(None, description="Max training samples (None for all)")

# Input schema for prediction
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
    training_timestamp: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    training_required: bool
    last_trained: Optional[str] = None

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_version: str
    training_duration: float
    model_metrics: Optional[Dict[str, float]] = None
    artifacts_path: str

# Training endpoint
@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest, config: TrainingConfig = None):
    """
    Train or retrain the ETA prediction model.
    
    This endpoint:
    1. Validates if training is needed
    2. Executes the complete ML pipeline
    3. Loads the newly trained model into memory
    4. Returns training results and metrics
    """
    global model_artifacts, is_model_trained, training_lock
    
    if training_lock:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Training is already in progress. Please wait for it to complete."
        )
    
    # Check if training is needed
    if is_model_trained and not request.force_retrain:
        current_model_path = MODEL_PATH
        if os.path.exists(current_model_path):
            return TrainingResponse(
                status="skipped",
                message="Model already trained and available. Use force_retrain=True to retrain.",
                model_version="1.0.0",
                training_duration=0.0,
                artifacts_path=current_model_path
            )
    
    training_lock = True
    start_time = datetime.now()
    
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
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Update global paths to new artifacts
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        PIPELINE_PATH = data_transformation_config.transformed_object_file_path
        MODEL_PATH = "final_model/model.pkl"
        
        model_artifacts['pipeline'] = joblib.load(PIPELINE_PATH)
        model_artifacts['model'] = joblib.load(MODEL_PATH)
        is_model_trained = True
        
        logger.info(f"Model training completed successfully in {training_duration:.2f} seconds")
        
        # Get model path for artifacts
        artifacts_base = os.path.dirname(PIPELINE_PATH)
        artifacts_path = os.path.dirname(artifacts_base)
        
        return TrainingResponse(
            status="success",
            message="Model trained and loaded successfully",
            model_version=f"1.0.{int(datetime.now().timestamp())}",
            training_duration=training_duration,
            model_metrics={
                'r2_score': float(model_score),
                'accuracy': float(model_accuracy_score) if model_accuracy_score else None,
                'training_time_seconds': training_duration
            },
            artifacts_path=str(artifacts_path)
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    finally:
        training_lock = False

# Upload training data endpoint
@app.post("/train/upload-data", tags=["Training"])
async def upload_training_data(file: UploadFile = File(..., description="CSV or ZIP file containing training data")):
    """
    Upload training data file and trigger training.
    
    Supports:
    - Single CSV files (train_data.csv, test_data.csv, or single dataset)
    - ZIP files containing multiple CSV files
    """
    global training_lock
    
    if training_lock:
        raise HTTPException(
            status_code=429,
            detail="Training is already in progress. Please wait."
        )
    
    if not file.filename.endswith(('.csv', '.zip')):
        raise HTTPException(
            status_code=400,
            detail="Only CSV or ZIP files are supported"
        )
    
    temp_dir = None
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = file.filename
            file_path = os.path.join(temp_dir, filename)
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Handle ZIP files
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find CSV files in extracted content
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise HTTPException(status_code=400, detail="No CSV files found in ZIP")
                
                # Place CSV files in expected data directory structure
                data_dir = os.path.join(temp_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                for csv_file in csv_files:
                    shutil.move(
                        os.path.join(temp_dir, csv_file),
                        os.path.join(data_dir, csv_file)
                    )
            else:
                # Handle single CSV file
                data_dir = os.path.join(temp_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                shutil.move(file_path, os.path.join(data_dir, "train_data.csv"))
            
            # Update config to use uploaded data
            training_pipeline_config = TrainingPipelineConfig()
            training_pipeline_config.artifact_root = temp_dir
            
            # Trigger training with uploaded data
            training_request = TrainingRequest(force_retrain=True)
            training_config = TrainingConfig()
            
            # Manually call training logic
            result = await train_model(training_request, training_config)
            return result
            
    except Exception as e:
        logger.error(f"Data upload and training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload and training failed: {str(e)}")
    
    finally:
        if temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    model_loaded = (
        'pipeline' in model_artifacts 
        and model_artifacts['pipeline'] is not None
        and 'model' in model_artifacts
        and model_artifacts['model'] is not None
    )
    
    training_required = not is_model_trained or not model_loaded
    
    # Try to get last trained timestamp
    last_trained = None
    if os.path.exists(MODEL_PATH):
        try:
            # Get timestamp of model file
            last_trained = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        message="ETA Prediction API is running" if model_loaded else "Model not loaded - training required",
        model_loaded=model_loaded,
        training_required=training_required,
        last_trained=last_trained
    )

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_eta(input_data: DeliveryInput):
    """Predict ETA using the loaded model"""
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
            model_version="1.0.0",
            training_timestamp=None  # Could store this in a config file
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(input_data: list[DeliveryInput]):
    """Batch prediction endpoint"""
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

# Model information endpoint
@app.get("/model/info", tags=["Model"])
def get_model_info():
    """Get information about the loaded model"""
    global model_artifacts, is_model_trained
    
    if not is_model_trained or 'pipeline' not in model_artifacts or 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not trained. Please call /train endpoint first.")
    
    # Get last trained timestamp
    last_trained = None
    if os.path.exists(MODEL_PATH):
        try:
            last_trained = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        except:
            pass
    
    return {
        "model_version": "1.0.0",
        "model_type": type(model_artifacts['model']).__name__,
        "pipeline_type": type(model_artifacts['pipeline']).__name__,
        "last_trained": last_trained,
        "status": "ready"
    }

# Training status endpoint
@app.get("/training/status", tags=["Training"])
def get_training_status():
    """Get current training status"""
    global training_lock, is_model_trained
    
    model_loaded = (
        'pipeline' in model_artifacts 
        and model_artifacts['pipeline'] is not None
        and 'model' in model_artifacts
        and model_artifacts['model'] is not None
    )
    
    return {
        "is_trained": is_model_trained,
        "model_loaded": model_loaded,
        "training_in_progress": training_lock,
        "training_required": not is_model_trained or not model_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
