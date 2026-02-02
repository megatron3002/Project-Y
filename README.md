# Explainable AI Visual Search Engine

An end-to-end visual search engine that allows users to upload an image of a product, finds similar items, and displays a Grad-CAM heatmap overlay to explain the similarity.

## Project Structure
- `app/`: FastAPI application code.
- `ml_engine/`: Machine learning logic (model, FAISS index, feature extraction).
- `frontend/`: Streamlit frontend (or React).
- `notebooks/`: Jupyter notebooks for experiments.

## Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t visual-search .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8080:8080 visual-search
   ```

3. **Access the API:**
   - Go to `http://localhost:8080/docs` to view the interactive API documentation (Swagger UI).
   - Health check: `http://localhost:8080/health`

## Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
