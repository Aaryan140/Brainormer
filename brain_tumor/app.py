import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the classes
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Function to perform detection and plot results
def detect_and_plot(image, model):
    results = model.predict(image)[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    
    detected_class = None
    
    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"{classes[int(cls)]} {conf:.2f}", color='white', fontsize=12, backgroundcolor='red')
        
        if conf > 0.5:  # Assuming a confidence threshold of 0.5
            detected_class = classes[int(cls)]
    
    plt.axis('off')
    
    # Save the plot to a BytesIO object to display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf, detected_class

# Function to get response from Gemini model

# Function to autogenerate based on what type of tumor
def get_gemini_response(tumor_type):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""Provide a brief overview of the {tumor_type} brain tumor. Include the following information:
    1. Brief description
    2. Common symptoms
    3. Typical treatment options
    
    Format the response in markdown with appropriate headers."""
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to handle user queries using gemini api
def get_gemini_response_for_query(user_query):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"{user_query}"
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit app setup
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Brain Tumor Detector</h1>", unsafe_allow_html=True)

st.subheader("Upload Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image using PIL
    image = Image.open(uploaded_image)
    
    # Convert image to RGB format
    image = image.convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to a format suitable for YOLO model
    image_np = np.array(image)
    
    # Load the YOLO model
    model_path = "BRAIN_TUMOR_DETECTOR_model.pt"  # Update this path to your model
    model = load_model(model_path)
    
    if model is not None:
        # Perform detection and get the result plot
        result_plot, detected_tumor_type = detect_and_plot(image_np, model)
        
        # Display the result plot in Streamlit
        st.image(result_plot, caption='Detection Results')
        
        if detected_tumor_type and detected_tumor_type != "No Tumor":
            st.subheader(f"Detected Tumor Type: {detected_tumor_type}")
            
            # Generate and display information about the detected tumor type
            tumor_info = get_gemini_response(detected_tumor_type)
            st.markdown(tumor_info)
        elif detected_tumor_type == "No Tumor":
            st.subheader("No tumor detected in the image.")
        else:
            st.subheader("No confident detection made. Please try with a clearer image.")

# Add user query section
st.subheader("Ask any query about brain tumors:")
user_query = st.text_area("Enter your query here:")

if user_query:
    response = get_gemini_response_for_query(user_query)
    st.markdown(response)