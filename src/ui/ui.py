import streamlit as st
import requests

st.set_page_config(page_title="Ensemble Embedding Viewer", page_icon="🧠")

st.title("🧠 Ensemble Embedding Viewer")
st.markdown("Compare how different embedding models interpret the same text relative to a pre-loaded dataset.")

# 1. Fetch the available datasets from the Kubernetes API
try:
    datasets_response = requests.get("http://localhost:8000/datasets", timeout=5)
    available_datasets = datasets_response.json().get("available_datasets", [])
except:
    available_datasets = []

if not available_datasets:
    st.error("🚨 Could not load datasets. Is the K8s port-forward running?")

# 2. Create a dropdown to select the dataset
selected_dataset = st.selectbox("Select a Dataset:", available_datasets)

# 3. Text input for the query
query = st.text_input("Enter your search query:", "Which brand is known for environmental sustainability?")

# 4. Submit button
if st.button("Compare Models", type="primary", use_container_width=True):
    if not query or not selected_dataset:
        st.warning("Please enter a query and select a dataset.")
    else:
        with st.spinner(f"Querying models using the '{selected_dataset}' dataset..."):
            try:
                response = requests.post(
                    "http://localhost:8000/compare-all-db",
                    json={"dataset_name": selected_dataset, "query": query},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    st.error(f"API Error: {data['error']}")
                else:
                    st.subheader(f"Results for: *'{data['query']}'* in **{data['dataset_used']}**")
                    
                    model_names = list(data["ensemble_comparison"].keys())
                    cols = st.columns(len(model_names))
                    
                    # draw model columns
                    for idx, model_name in enumerate(model_names):
                        with cols[idx]:
                            result = data["ensemble_comparison"][model_name]
                            st.markdown(f"### {model_name}")
                            
                            if "error" in result:
                                st.error(f"**Error:** {result['error']}")
                            else:
                                st.metric(label="Top Match", value=result["top_category"])
                                st.progress(min(result["score"] / 1.0, 1.0))
                                st.write(f"**Cosine Score:** `{result['score']:.4f}`")
                                # number vector display
                                with st.expander("🧮 View Raw Vectors"):
                                    vector_data = result.get("vector_preview", [])
                                    vector_size = result.get("vector_size", "Unknown")
                                    if vector_data:
                                        formatted_vector = [round(v,4) for v in vector_data]
                                        st.write(f"**Category Vector (first 8 dims out of {vector_size} dims)**")
                                        st.code(str(formatted_vector), language="python")
                                    else:
                                        st.write("No vector preview available.")
                            
            except requests.exceptions.RequestException as e:
                st.error("🚨 **Could not connect to the Ensemble API.** Make sure your K8s port-forward is running!")
