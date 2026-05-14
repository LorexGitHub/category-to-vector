from fastapi import FastAPI
from pydantic import BaseModel
import os
from sentence_transformers import util

app = FastAPI(title="Embedding Model API")

MODEL_NAME = os.getenv("MODEL_NAME", "granite")

if MODEL_NAME == "granite":
    from granite_model import encode_categories, encode_query
elif MODEL_NAME == "qwen3":
    from qwen3_model import encode_categories, encode_query
elif MODEL_NAME == "nemotron":
    from nemotron_model import encode_categories, encode_query
elif MODEL_NAME == "jina":
    from jina_model import encode_categories, encode_query
elif MODEL_NAME == "harrier":
    from harrier_model import encode_categories, encode_query
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")


class CompareRequest(BaseModel):
    categories: list[str]
    query: str


@app.post("/compare")
async def compare(request: CompareRequest):
    category_embeddings = encode_categories(request.categories)
    query_embedding = encode_query(request.query)

    cos_scores = util.cos_sim(query_embedding, category_embeddings)[0]

    results = []
    for idx, (cat, score) in enumerate(zip(request.categories, cos_scores.tolist())):
        results.append({
            "category": cat, 
            "score": score,
            "vector_preview": category_embeddings[idx][:8].tolist(),
            "vector_size": len(category_embeddings[idx])
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {"model": MODEL_NAME, "results": results}
