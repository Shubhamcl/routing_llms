import os
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModel
from torch import nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================
# Model Architecture (from training notebook)
# ============================

class CompactQualityClassifier(nn.Module):
    """Compact transformer-based quality classifier optimized for limited data."""
    
    def __init__(self, transformer_model_name: str, dropout_rate: float = 0.3, 
                 freeze_transformer: bool = False):
        super(CompactQualityClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        
        # Freeze transformer parameters if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Transformer layers frozen - only using classification head")
        else:
            logger.info("All transformer layers loaded")
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Simplified architecture for smaller dataset
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, sentences: list) -> torch.Tensor:
        # Tokenize and encode as batch
        encoding = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(next(self.transformer.parameters()).device)
        
        outputs = self.transformer(**encoding)
        # Use the [CLS] token embedding for classification
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification layers
        x = self.dropout(pooled)
        logits = self.classifier(x)
        
        return logits

# ============================
# API Models (Pydantic)
# ============================

class TextInput(BaseModel):
    """Single text input for quality prediction."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to evaluate for quality")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchTextInput(BaseModel):
    """Batch text input for quality prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to evaluate")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        validated_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty or whitespace only')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} is too long (max 10,000 characters)')
            validated_texts.append(text.strip())
        
        return validated_texts

class RoutePrediction(BaseModel):
    """Route prediction result."""
    text: str = Field(..., description="Original input text")
    score: float = Field(..., ge=0.0, le=1.0, description="Sigmoid output (0=poor, 1=good)")
    route_label: str = Field(..., description="Model to route to (Weak/Strong model)")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")

class BatchRoutePrediction(BaseModel):
    """Batch route prediction result."""
    predictions: List[RoutePrediction] = Field(..., description="List of route predictions")
    total_processing_time_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
    average_processing_time_ms: float = Field(..., ge=0.0, description="Average processing time per text")

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    device: str = Field(..., description="Device being used")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

# ============================
# Model Manager
# ============================

class ModelManager:
    """Manages model loading, inference, and state."""
    
    def __init__(self):
        self.model: Optional[CompactQualityClassifier] = None
        self.device: Optional[torch.device] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.startup_time: float = 0.0
        
    def load_model(self, model_path: str, config_path: Optional[str] = None) -> None:
        """Load the trained model and configuration."""
        try:
            import time
            start_time = time.time()
            
            logger.info(f"Loading model from: {model_path}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Default model parameters (can be overridden by config)
            transformer_model_name = 'distilbert-base-uncased'
            dropout_rate = 0.3
            freeze_transformer = False
            
            # Override with config if available
            if self.model_config:
                transformer_model_name = self.model_config.get('config', {}).get('transformer_model', transformer_model_name)
                dropout_rate = self.model_config.get('config', {}).get('dropout_rate', dropout_rate)
                freeze_transformer = self.model_config.get('config', {}).get('freeze_transformer', freeze_transformer)
            
            # Initialize model
            self.model = CompactQualityClassifier(
                transformer_model_name=transformer_model_name,
                dropout_rate=dropout_rate,
                freeze_transformer=freeze_transformer
            )
            
            # Load trained weights
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.startup_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.startup_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_route(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict route for a list of texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        import time
        start_time = time.time()
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                outputs = self.model(texts)
                
                # Convert to probabilities
                probabilities = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                
                # Handle single prediction
                if isinstance(probabilities, np.ndarray) and probabilities.ndim == 0:
                    probabilities = [float(probabilities)]
                else:
                    probabilities = probabilities.tolist()
                
                # Convert to quality scores and labels
                results = []
                processing_time = (time.time() - start_time) * 1000  # ms
                
                for i, (text, prob) in enumerate(zip(texts, probabilities)):
                    # Note: Model outputs 0 for good quality/weak model, 1 for poor quality/strong model
                    # So we invert the probability for quality score
                    score = prob
                    label = "Weak Model" if score <= 0.5 else "Strong Model"
                    results.append({
                        "text": text,
                        "score": score,
                        "route_label": label,
                        "processing_time_ms": processing_time / len(texts)
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# ============================
# FastAPI Application
# ============================

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Router API service...")
    
    # Look for model files
    model_path = os.getenv("MODEL_PATH", "./runs")
    config_path = os.getenv("CONFIG_PATH", "./config.json")
    
    # Find the latest model file if MODEL_PATH is a directory
    if Path(model_path).is_dir():
        # model_files = list(Path(model_path).glob("best_model_*.pt"))
        model_files = list(Path(model_path).glob("best_model_slow_lr_20250813_170739.pt"))
        if model_files:
            # Get the most recent model file
            model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
            logger.info(f"Found model file: {model_path}")
        else:
            logger.warning("No model files found in directory")
    
    # Load model if available
    if Path(model_path).exists():
        try:
            model_manager.load_model(model_path, config_path if Path(config_path).exists() else None)
            logger.info("Model loaded successfully during startup")
        except Exception as e:
            logger.error(f"Failed to load model during startup: {e}")
    else:
        logger.warning(f"Model file not found: {model_path}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Router API service...")

# Create FastAPI app
app = FastAPI(
    title="Router Quality Classifier API",
    description="API for predicting text quality using a transformer-based classifier",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# API Endpoints
# ============================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Router Quality Classifier API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    import time
    uptime = time.time() - model_manager.startup_time if model_manager.startup_time > 0 else 0
    
    return HealthCheck(
        status="healthy" if model_manager.model is not None else "no_model",
        model_loaded=model_manager.model is not None,
        device=str(model_manager.device) if model_manager.device else "unknown",
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=RoutePrediction)
async def predict_route(input_data: TextInput):
    """Predict route for a single text."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = model_manager.predict_route([input_data.text])
        return RoutePrediction(**results[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchRoutePrediction)
async def predict_route_batch(input_data: BatchTextInput):
    """Predict route for multiple texts."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        results = model_manager.predict_route(input_data.texts)
        
        total_time = (time.time() - start_time) * 1000  # ms
        avg_time = total_time / len(input_data.texts)

        predictions = [RoutePrediction(**result) for result in results]

        return BatchRoutePrediction(
            predictions=predictions,
            total_processing_time_ms=total_time,
            average_processing_time_ms=avg_time
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ============================
# Main
# ============================

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "router_api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False
    )
