from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        # Jina requires trust_remote_code=True
        _model = SentenceTransformer(
            "jinaai/jina-embeddings-v5-text-small",
            trust_remote_code=True
        )
    return _model

def encode_categories(categories):
    model = get_model()
    # Use task="text-matching" for category similarity
    return model.encode(categories, task="text-matching")

def encode_query(query):
    model = get_model()
    # Wrap query in list for consistent return shape
    return model.encode(query, task="text-matching")

if __name__ == "__main__":
    from input import categories, query
    from sentence_transformers import util
    import torch

    print(f"[Jina Model]\n")

    embeddings = encode_categories(categories)
    query_emb = encode_query(query)

    cos_scores = util.cos_sim(query_emb, embeddings)[0]

    print(f"Query: \"{query}\"\n")
    print(f"{'Category':<12} | {'Score':<8} | Vector (first 8 dims)")
    print("-" * 60)
    
    # Sort by score descending
    results = sorted(zip(categories, cos_scores.tolist(), embeddings), key=lambda x: x[1], reverse=True)
    
    for cat, score, emb in results:
        print(f"{cat:<12} | {score:.4f}  | {emb[:8].tolist()}")