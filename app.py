# app_multi_disease.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Multi-Disease Lung Detector",
    page_icon="🫁",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the pre-trained model - UPDATED FILENAME
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lungs_disease_vgg16.h5')  # Updated filename
    return model

model = load_model()

# Class names for 4-class model
class_names = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

# Sidebar
st.sidebar.title("🛠️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.7)

# Main app
st.title(" Multi-Disease Lung Detection System")
st.markdown("""
Detect **Pneumonia, COVID-19, and Tuberculosis** from chest X-ray images.
Upload an X-ray for comprehensive analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the uploaded image
        original_img = Image.open(uploaded_file).convert("RGB")
        st.image(original_img, caption='Uploaded X-ray Image', use_column_width=True)
        
        # Preprocess the image for the model
        img = original_img.resize((224, 224))
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        elif img_array.shape[-1] == 1:
            img_array = np.stack([img_array.squeeze()] * 3, axis=-1)
        
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

        # Make prediction
        with st.spinner('🔬 Analyzing for multiple diseases...'):
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_disease = class_names[predicted_class_idx]

        # Display results
        st.subheader("🔍 Analysis Results")
        
        if confidence < confidence_threshold:
            st.warning("⚠️ Low confidence prediction. Please consult a doctor.")
        
        # Color-coded results
        result_text = f"**Prediction:** {predicted_disease}"
        confidence_text = f"**Confidence:** {confidence * 100:.2f}%"

        if predicted_disease == "Normal":
            st.success(result_text)
            st.info(confidence_text)
        else:
            st.error(result_text)
            st.warning(confidence_text)

        # Medical advice
        medical_advice = {
            "Normal": "🧑‍⚕️ No abnormalities detected. Regular check-ups recommended.",
            "Pneumonia": "🧑‍⚕️ Pneumonia detected. Please consult a pulmonologist. Antibiotics may be required.",
            "COVID-19": "🦠 COVID-19 signs detected. Isolation recommended. Consult doctor immediately.",
            "Tuberculosis": "⚠️ Tuberculosis suspected. Urgent specialist consultation required. Contagious risk."
        }
        st.info(f"**Medical Advice:** {medical_advice[predicted_disease]}")

    with col2:
        # Disease probability dashboard
        st.subheader("📊 Disease Probability Dashboard")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(class_names))
        colors = ['#4CAF50', '#FF9800', '#9C27B0', '#F44336']  # Green, Orange, Purple, Red
        
        bars = ax.barh(y_pos, predictions[0], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Multi-Disease Detection Results')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            risk_level = "Low" if predictions[0][i] < 0.3 else "Medium" if predictions[0][i] < 0.7 else "High"
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{predictions[0][i]:.3f} ({risk_level})', 
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        st.pyplot(fig)
        
        # Detailed probabilities table
        st.subheader("📋 Detailed Probabilities")
        prob_data = []
        for i, disease in enumerate(class_names):
            prob_data.append({
                'Disease': disease,
                'Probability': f'{predictions[0][i]:.4f}',
                'Risk Level': 'Low' if predictions[0][i] < 0.3 else 'Medium' if predictions[0][i] < 0.7 else 'High'
            })
        
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)

    # Add to history
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': uploaded_file.name,
        'prediction': predicted_disease,
        'confidence': f"{confidence * 100:.2f}%",
        'all_probabilities': predictions[0].tolist()
    }
    st.session_state.history.append(history_entry)

    st.success("✅ Analysis complete!")

# History Section
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Prediction History")

if st.session_state.history:
    st.sidebar.write("**Recent predictions:**")
    for i, entry in enumerate(reversed(st.session_state.history[-3:])):
        emoji = "✅" if entry['prediction'] == "Normal" else "⚠️"
        st.sidebar.markdown(f"""
        {emoji} **{entry['timestamp'].split()[1]}**  
        `{entry['filename'][:15]}...` → **{entry['prediction']}** ({entry['confidence']})
        """)
    
    # Export functionality
    if st.sidebar.button("📤 Export History to CSV"):
        df = pd.DataFrame(st.session_state.history)
        # Clean up the dataframe for export
        export_df = df[['timestamp', 'filename', 'prediction', 'confidence']].copy()
        csv = export_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="lung_disease_predictions.csv",
            mime="text/csv"
        )
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.sidebar.success("History cleared!")
        st.rerun()
else:
    st.sidebar.info("No prediction history yet. Upload an image to get started!")

# Footer with disclaimer
st.markdown("---")
st.warning("""
**⚠️ Important Disclaimer:**  
This tool is for educational purposes only developed for Reasearch & Minor project Only. It is **NOT** a certified medical device.
**Always consult a qualified healthcare professional for diagnosis and treatment.**
Do not make medical decisions based solely on this tool's output.
""")
