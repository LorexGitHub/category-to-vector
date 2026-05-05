from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
        "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,
        device="cpu",
        model_kwargs={
            "attn_implementation": "eager", 
            "torch_dtype": "bfloat16"
        }
)

def encode_categories(categories):
    model = get_model()
    return model.encode_document(categories)

def encode_query(query):
    model = get_model()
    return model.encode_query(query)

if __name__ == "__main__":
    from input import categories, query
    from sentence_transformers import util
    import torch

    print(f"[Nemotron Model]\n")

    embeddings = encode_categories(categories)
    query_emb = encode_query(query)

    cos_scores = util.cos_sim(query_emb, embeddings)[0]

    print(f"Query: \"{query}\"\n")
    print(f"{'Category':<12} | {'Score':<8} | Vector (first 8 dims)")
    print("-" * 60)
    results = sorted(zip(categories, cos_scores.tolist(), embeddings), key=lambda x: x[1], reverse=True)
    for cat, score, emb in results:
        print(f"{cat:<12} | {score:.4f}  | {emb[:8].tolist()}")