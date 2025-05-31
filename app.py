import streamlit as st
from model.predict import predict
import logging 

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log', mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)  # <== This line is crucial!

logger.info("This should go to the file")

# Set up page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🕵️‍♂️ Are Your News Fake?")
st.subheader("Let's use TWO models to figure out!")

st.sidebar.header("Configurations")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"
def get_confidence_description(confidence):
    """Get human-readable confidence description"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.7:
        return "Medium-High"
    elif confidence >= 0.6:
        return "Medium"
    elif confidence >= 0.5:
        return "Low-Medium"
    else:
        return "Low"    

model_options = ["lr", "nb"]  # Add youQr available models
vectorizer_options = ["tfidf", "count"]

selected_model = st.sidebar.selectbox(
        "Select Model:",
        model_options,
        index=0,
        help="Choose the machine learning model for classification"
    )
    
selected_vectorizer = st.sidebar.selectbox(
        "Select Vectorizer:",
        vectorizer_options,
        index=0,
        help="Choose the text vectorization method"
    )
col1, col2 = st.columns([2, 1])


with col1:
    st.header("Input your news:")
    title_input = st.text_input("Title Of The News Article",placeholder="Enter a title for your text")
    text_input = st.text_area("News Article",height=200, placeholder="Enter the text you want to classify here...")
    predict_button = st.button("Let's know if it's fake or not", type="primary", use_container_width=True)
with col2:
    st.header("Results")
    if predict_button and text_input.strip():
            with st.spinner("Classifying text..."):
                    prediction, confidence = predict(
                        text_input, 
                        selected_model, 
                        selected_vectorizer
                    )
            logger.info(f"Prediction: {prediction} with confidence {confidence:.3f}")
            logger.info(f"Input processed - Title: '{title_input[:30]}...', Text length: {len(text_input)} chars.")
            logger.info(f'Model: {selected_model}, Vectorizer: {selected_vectorizer}')    
            st.success("Classification Complete!")
            st.metric(
                    label="Predicted Class",
                    value=prediction,
                    delta=f"{confidence:.1%} confidence"
                )
