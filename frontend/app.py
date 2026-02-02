import streamlit as st
import requests
from PIL import Image
import io
import base64

# Configuration
API_URL = "http://localhost:8080" # Assumes backend is running locally or mapped port

st.set_page_config(page_title="XAI Visual Search", layout="wide")

st.title("🔍 Explainable AI Visual Search Engine")
st.markdown("Upload a product image to find similar items and see **why** they match.")

# Sidebar
st.sidebar.header("Options")
show_heatmap = st.sidebar.checkbox("Show AI Attention (Grad-CAM)", value=False)

# File Upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display query
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Query Image")
        st.image(uploaded_file, use_column_width=True)
    
    # Search Button
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                # Send to API
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/search", files=files)
                
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Found {len(results)} matches.")
                    
                    # Display results
                    st.subheader("Top Matches")
                    
                    cols = st.columns(len(results))
                    
                    for i, res in enumerate(results):
                        with cols[i]:
                            # Get Image Path from metadata (Since we are local/docker volume shared)
                            # In a real deployed web app, the backend would return a URL to the image (S3/GCS)
                            # Here we might need to handle the path mapping if frontend and backend are separate.
                            # Assuming for this demo we are effectively local or sharing volume data.
                            
                            # However, Streamlit cannot access the backend's file system directly if they are separate containers
                            # without volume mount.
                            # For simplicity, we assume we can read the file path if running locally.
                            # Or we can ask the backend to serve the image.
                            # Let's assume the backend serves images or we just rely on the 'path' being accessible.
                            
                            # Actually, Streamlit won't be able to display an absolute path from the backend if it's in a container.
                            # But let's proceed with the happy path that we can read it, or maybe we should have an image serving endpoint.
                            # For now, let's assume the user is running this locally.
                            
                            img_path = res['metadata']['path']
                            
                            try:
                                if show_heatmap:
                                    # Fetch explanation
                                    exp_res = requests.post(
                                        f"{API_URL}/explain", 
                                        params={"image_path": img_path}
                                    )
                                    if exp_res.status_code == 200:
                                        b64_img = exp_res.json()['heatmap_base64']
                                        img_data = base64.b64decode(b64_img)
                                        st.image(img_data, caption=f"Score: {res['score']:.2f}", use_column_width=True)
                                    else:
                                        st.error("Explainer failed")
                                else:
                                    st.image(img_path, caption=f"Score: {res['score']:.2f}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Could not load image: {img_path}")
                            
                            st.caption(f"{res['metadata']['filename']}")
                            
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
