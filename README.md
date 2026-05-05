# 🧠 Category-to-Vector: An Ensemble Embedding API
A microservices-based API for benchmarking and comparing how different state-of-the-art embedding models map categorical text to feature vectors, deployed via Kubernetes.

### Demo




https://github.com/user-attachments/assets/b8ce2cf8-4d3c-484c-b727-31c9ba45409f




### 🎯 The Problem
In Natural Language Processing, different embedding models capture different semantic nuances. An e-commerce search query like "apple" might map closer to "fruit" in one model, and closer to "laptops" in another. Relying on a single model can create blind spots in search engines or recommendation systems.

### 💡 The Solution
This project provides an Ensemble Embedding API that runs up to 5 (currently 3 and more are planned) distinct embedding models simultaneously as isolated Kubernetes pods. By sending a single query to the ensemble, you can instantly compare how each model interprets the text relative to a pre-loaded database of categories, allowing you to choose the best model for your specific domain.

### 🤖 The Models 
This project leverages sentence-transformers to host the following models: 

     IBM Granite: https://huggingface.co/ibm-granite/granite-embedding-small-english-r2
     Qwen: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
     Nemotron: https://huggingface.co/nvidia/llama-embed-nemotron-8b
     Jina: https://huggingface.co/jinaai/jina-embeddings-v5-text-small
     Harrier: https://huggingface.co/microsoft/harrier-oss-v1-270m
     

### 📁 Project Structure
```
├── src/
│   ├── models/          # Individual embedding models
│   │   ├── granite_model.py
│   │   ├── qwen3_model.py
│   │   ├── nemotron_model.py
│   │   ├── jina_model.py
│   │   └── harrier_model.py
│   ├── api/             # FastAPI applications
│   │   ├── api.py       # Model API (per-model container)
│   │   └── ensemble_api.py  # Ensemble API (aggregates all models)
│   └── ui/              # Streamlit frontend
│       └── ui.py
├── data/
│   └── datasets.json    # Category datasets
├── infra/
│   ├── k8s/             # Kubernetes manifests
│   └── aws/             # AWS EKS configuration
├── Dockerfile           # Model service image
├── Dockerfile.ensemble  # Ensemble API image
├── docker-compose.yaml  # Local development
├── requirements.txt
└── ensemble_requirements.txt
```

### ⚙️ Tech Stack

    Language: Python 3.9
    API Framework: FastAPI + Uvicorn
    ML Library: Sentence-Transformers, PyTorch (CPU-optimized)
    Frontend: Streamlit
    Data Persistence: JSON + Kubernetes ConfigMaps
    Containerization: Docker
    Orchestration: Kubernetes (Kind / AWS EKS)
    CI/CD: GitHub Actions
     

### 🚀 Getting Started (Local)
Prerequisites

    Docker Desktop installed
    kind (Kubernetes in Docker) installed
    kubectl installed
    Python 3.9+ installed (for the Streamlit UI)


Clone the repository and build the microservices:
1. Create the Local Kubernetes Cluster
```
kind create cluster --name ml-ensemble
```
 
2. Build and Load Docker Images
Because Kind runs its own internal Docker engine, we must build our images and load them into the cluster: 
```
# Build the model API image
docker build -t lorexdocker/embedding-service:latest .

# Build the Ensemble API image
docker build -t lorexdocker/ensemble-api:latest -f Dockerfile.ensemble .

# Load both into Kind
kind load docker-image lorexdocker/embedding-service:latest --name ml-ensemble
kind load docker-image lorexdocker/ensemble-api:latest --name ml-ensemble
```
⏳ Wait! Do not proceed until you see loaded successfully! for Granite, Harrier, and Jina in your terminal. The models need time to download and load into memory. 

3. Inject the Datasets via ConfigMap
We use a K8s ConfigMap to hold our datasets.json. This allows us to update the data without rebuilding Docker images!

```
kubectl create configmap datasets-config --from-file=data/datasets.json
```

4. Deploy to Kubernetes!
Apply the K8s manifests to spin up the models and the ensemble API:
```
kubectl apply -f infra/k8s/
```
Wait for the pods to spin up and download the model weights (this may take a few minutes):
```
kubectl get pods
```

5. Expose the API via Port-Forward
In a separate terminal, forward port 8000 to the Ensemble API (leave this running):
```
kubectl port-forward service/ensemble 8000:8000
```

6. Launch the Streamlit UI
Open a third terminal window and start the visual UI:

    Mac/Linux:

```
streamlit run src/ui/ui.py
```


    Windows (if 'streamlit' is not recognized):


```
python -m streamlit run src/ui/ui.py
```

7. Compare Models!
```
1. A browser window will open to http://localhost:8501. 
2. Select a dataset from the dropdown (e.g., "all", "mythology_creatures", "cars"). 
3. Type a search query (e.g., "A programming language named after a snake"). 
4. Hit Compare Models and watch the K8s-hosted models return their scores in real-time! 
```
### 🗂️ Updating the Datasets (Zero Downtime)
If you add new categories to datasets.json, you can update the live Kubernetes app without rebuilding Docker: 
1. Delete the old config: 
```kubectl delete configmap datasets-config ```

2. Create the new one: 
```kubectl create configmap datasets-config --from-file=data/datasets.json ```

3. Restart the pod: 
```kubectl rollout restart deployment ensemble-api ```

### ☁️ Cloud Deployment (AWS EKS)

This application is cloud-agnostic and has been tested on AWS EKS. The Kubernetes manifests in the infra/k8s/ folder apply identically to local clusters and AWS.
AWS Infrastructure

The cluster configuration is defined in infra/aws/cluster-config.yaml using eksctl.

### ⚠️ Note on AWS Free Tier & ML Models
AWS Free Tier instances (t3.micro - 1GB RAM) are fundamentally incompatible with PyTorch-based ML models. During testing on AWS, containers request 1GB+ of memory just to load the model weights. On a t3.micro, this results in Kubernetes scheduling failures (Insufficient memory) or OOMKilled errors. 

A t3.xlarge instance (16GB RAM) is required to run the ensemble in production. (See screenshots in the /assets folder for AWS node deployment and memory constraint diagnostics).
How to deploy to AWS EKS:

    Ensure eksctl and awscli are installed and configured.
    Create the cluster: eksctl create cluster -f infra/aws/cluster-config.yaml
    Connect kubectl: aws eks update-kubeconfig --name ml-ensemble-cluster --region eu-central-1
    Deploy the app: kubectl apply -f infra/k8s/
    CRITICAL: Destroy the cluster when done to avoid charges: `eksctl delete cluster -f infra/aws/cluster-config.yaml
