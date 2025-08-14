# Routing LLMs

This repo goes through three routing methods and evaluates them over a small dataset. Additionally it creates small dataset of queries, generates responses from LLM apis (using openrouter and Groq).

We explore semantically routing queries at first, then using a LLM to route queries and finally we train a BERT-like router.

The LLM based solution out-performs over the dataset used. Just to demonstrate the ability, the trained router BERT-like model is provided with a API using FastAPI.


### TL;DR aim of smart routing:

- **Route simple queries** (factual, general knowledge) → **Weak Model** (e.g., Llama 3.1 8B)
- **Route complex queries** (coding, specialized) → **Strong Model** (e.g., GPT-OSS, Qwen3:coder)
- **To maintain quality** while reducing costs and improving speed

### Recommended Flow:
- Refer `notebook-report.ipynb` for non-finetuned routing.
- Refer `Dataset-notebook.ipynb` to understand how the data was created.
- Refer `router_training_notebook.ipynb` to see how the third router is trained.
- Then run the API to run it.


## Quick Overview

The system implements three routing approaches:

### 1. Semantic Similarity Routing (Part 1)
- Uses sentence transformers to encode queries
- Compares embeddings with predefined model personalities
- Routes based on cosine similarity scores

### 2. LLM-Based Classification (Part 2)
- Uses an LLM to classify query categories
- Rule-based routing based on content type
- Uses external API calls for classification

### 3. Trained Quality Classifier (Part 3)
- **Custom transformer-based binary classifier**
- Trained on response quality rated responses
- Predicts routing decisions, but overfits
- Model API using FastAPI included

### 4. Running API
- Dockerfile and docker-composed included
- Cancelled plans to host API on EC2
- Not sure when this was gonna be tested

## Installation

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Shubhamcl/routing_llms.git
cd routing_llms
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download trained model**
```bash
bash download_model.sh
```


### Docker Setup

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **For GPU support** (uncomment GPU section in docker-compose.yml)
```bash
docker-compose up -d router-api-gpu
```

## Usage

### Starting the API Server

**Local:**
```bash
python router_api.py
```

**Docker:**
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

### Making Predictions

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a Python function to calculate fibonacci numbers"}'
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is machine learning?", "Implement a binary search algorithm"]}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch text predictions |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Response Format

**Single Prediction Response:**
```json
{
  "text": "input query text",
  "score": 0.85,
  "route_label": "Strong Model",
  "processing_time_ms": 45.2
}
```

**Batch Prediction Response:**
```json
{
  "predictions": [
    {
      "text": "query 1",
      "score": 0.25,
      "route_label": "Weak Model",
      "processing_time_ms": 15.3
    }
  ],
  "total_processing_time_ms": 92.7,
  "average_processing_time_ms": 46.35
}
```

### Routing Logic
- **Score ≤ 0.5** → Route to **Weak Model** (sufficient for simple queries)
- **Score > 0.5** → Route to **Strong Model** (complex queries requiring more capability)
