from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import asyncio
import json
from pathlib import Path

app = FastAPI(title="Ensemble API")

MODEL_URLS = {
    "Granite": "http://granite:8000/compare",
    "Harrier": "http://harrier:8000/compare",
    "Jina": "http://jina:8000/compare",
}

# ==========================================
# LOAD EXAMPLE DATA FROM JSON FILE
# ==========================================
DATASETS = {}
data_file = Path("datasets.json")
if data_file.exists():
    with open(data_file, "r") as f:
        DATASETS = json.load(f)
        # --- NEW: Auto-generate the "all" dataset! ---
    all_categories = []
    for dataset_name, categories in DATASETS.items():
        all_categories.extend(categories)
    
    # Remove duplicates (in case any word appears in two datasets) and add it
    DATASETS["all"] = list(set(all_categories))

# ==========================================
# ENDPOINTS
# ==========================================

# 1. Get available datasets
@app.get("/datasets")
def get_datasets():
    return {"available_datasets": list(DATASETS.keys())}

# 2. Compare using a dataset from the JSON file
class CompareDBRequest(BaseModel):
    dataset_name: str
    query: str

@app.post("/compare-all-db")
async def compare_all_models_db(request: CompareDBRequest):
    # Find the categories in our JSON data
    categories = DATASETS.get(request.dataset_name)
    
    if not categories:
        return {"error": f"Dataset '{request.dataset_name}' not found."}

    # Send those categories to the models!
    payload = {"categories": categories, "query": request.query}
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for model_name, url in MODEL_URLS.items():
            tasks.append(client.post(url, json=payload, timeout=60.0))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    ensemble_results = {}
    for model_name, response in zip(MODEL_URLS.keys(), responses):
        if isinstance(response, Exception):
            ensemble_results[model_name] = {"error": str(response)}
            continue
        if response.status_code != 200:
            ensemble_results[model_name] = {"error": f"Model API Error: {response.status_code}"}
            continue
        try:
            data = response.json()
            top_pick = data["results"][0]
            ensemble_results[model_name] = {
                "top_category": top_pick["category"],
                "score": top_pick["score"],
                "vector_preview": top_pick.get("vector_preview", []),
                "vector_size": top_pick.get("vector_size")
            }
        except Exception as e:
            ensemble_results[model_name] = {"error": f"JSON Parse Error: {str(e)}"}
            
    return {
        "query": request.query,
        "dataset_used": request.dataset_name,
        "ensemble_comparison": ensemble_results
    }
    
# for dynamic dataset management
@app.get("/datasets/{dataset_name}")
def get_dataset_categories(dataset_name: str):
    categories = DATASETS.get(dataset_name)
    if categories is None:
        return {"error": f"Dataset '{dataset_name}' not found."}
    return {"dataset_name": dataset_name, "categories": categories}

class UpdateDatasetRequest(BaseModel):
    categories: list[str]

@app.post("/datasets/{dataset_name}")
def update_dataset(dataset_name: str, request: UpdateDatasetRequest):
    # Update the in-memory dictionary
    DATASETS[dataset_name] = request.categories
    
    # Save the changes back to the JSON file so they persist!
    try:
        with open("datasets.json", "w") as f:
            json.dump(DATASETS, f, indent=4)
    except Exception as e:
        return {"error": f"Failed to save to file: {str(e)}"}

    return {"message": f"Dataset '{dataset_name}' updated successfully", "categories": request.categories}