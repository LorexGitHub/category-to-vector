from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        # Preserving the specific model_kwargs from your snippet
        _model = SentenceTransformer(
            "microsoft/harrier-oss-v1-270m", 
            model_kwargs={"dtype": "auto"}
        )
    return _model

def encode_categories(categories):
    model = get_model()
    # Documents/categories do not need the prompt
    return model.encode(categories)

def encode_query(query):
    model = get_model()
    # Queries specifically use the "web_search_query" prompt
    return model.encode(query, prompt_name="web_search_query")

if __name__ == "__main__":
    from input import categories, query
    from sentence_transformers import util
    import torch

    print(f"[Harrier Model]\n")

    embeddings = encode_categories(categories)
    query_emb = encode_query(query)

    cos_scores = util.cos_sim(query_emb, embeddings)[0]

    print(f"Query: \"{query}\"\n")
    print(f"{'Category':<12} | {'Score':<8} | Vector (first 8 dims)")
    print("-" * 60)
    results = sorted(zip(categories, cos_scores.tolist(), embeddings), key=lambda x: x[1], reverse=True)
    for cat, score, emb in results:
        print(f"{cat:<12} | {score:.4f}  | {emb[:8].tolist()}")