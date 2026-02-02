import shutil
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import base64
import numpy as np
import cv2

# Import ML Engine
# Note: Ensure these are in PYTHONPATH or use relative imports structure properly
from ml_engine.model import FeatureExtractor
from ml_engine.index import VectorIndex
from ml_engine.xai import XAIEngine

app = FastAPI(title="Explainable AI Visual Search Engine", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
feature_extractor = None
vector_index = None
xai_engine = None

@app.on_event("startup")
async def load_models():
    global feature_extractor, vector_index, xai_engine
    print("Loading models...")
    feature_extractor = FeatureExtractor()
    xai_engine = XAIEngine()
    
    vector_index = VectorIndex()
    vector_index.load()
    print("Models loaded.")

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: dict

class ExplainResponse(BaseModel):
    image_id: str
    heatmap_base64: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "index_size": vector_index.index.ntotal if vector_index and vector_index.index else 0}

@app.post("/search", response_model=List[SearchResult])
async def search_image(file: UploadFile = File(...)):
    """
    Search for similar images.
    """
    if not feature_extractor or not vector_index:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Extract features
        query_vector = feature_extractor.extract(image)
        
        # Search
        results = vector_index.search(query_vector, k=5)
        
        response = []
        for i, (meta, score) in enumerate(results):
            # meta contains 'path', 'class', 'filename'
            # We use the path as ID for simplicity in this demo
            response.append(SearchResult(
                id=str(meta.get('path')), # In a real app this would be a database ID
                score=score,
                metadata=meta
            ))
            
        return response
        
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(image_path: str):
    """
    Generate Grad-CAM heatmap explanation for a specific image in the database.
    In a real app, we might pass the query image + target image ID.
    Here, we simplify: we explain why the model sees features in the *retrieved* image.
    Or, more commonly in visual search: "Why does this result look like my query?"
    
    Actually, standard Grad-CAM explains the class prediction. 
    For retrieval, we often overlay the heatmap on the result image to show what features (e.g. pattern, shape) were dominant.
    """
    if not xai_engine:
         raise HTTPException(status_code=503, detail="XAI Engine not loaded")
         
    try:
        # Load the image from disk (simulating database retrieval)
        # image_path comes from the search result metadata
        if not os.path.exists(image_path):
             raise HTTPException(status_code=404, detail="Image not found")
             
        img = Image.open(image_path).convert("RGB")
        
        # Generate heatmap
        # We don't have a specific target class index for retrieval "similarity",
        # but usually we want to see the dominant features. 
        # Passing None to target_class_idx uses the highest scroring class.
        overlay = xai_engine.explain(img, target_class_idx=None)
        
        # Convert overlay (numpy array) to base64 string for frontend
        overlay_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return ExplainResponse(image_id=image_path, heatmap_base64=img_str)
        
    except Exception as e:
        print(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
