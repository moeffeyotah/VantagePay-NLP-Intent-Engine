import streamlit as st
import joblib
import spacy
import os

# --- 1. RESOURCE INITIALIZATION ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load artifacts
    model = joblib.load(os.path.join(current_dir, "fanshawe_intent_svc_model.pkl"))
    tfidf = joblib.load(os.path.join(current_dir, "fanshawe_tfidf_vectorizer.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "fanshawe_label_encoder.pkl"))

    # Load spaCy engine
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except:
        import en_core_web_sm
        nlp = en_core_web_sm.load(disable=["parser", "ner"])

    return model, tfidf, encoder, nlp

model, tfidf, encoder, nlp = load_assets()

def lemmatize_text(text):
    doc = nlp(str(text).lower())
    return " ".join(
        [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    )

# --- 2. USER INTERFACE ARCHITECTURE ---
st.set_page_config(
    page_title="VantagePay Intent Engine", 
    page_icon="🌐", 
    layout="centered"
)

# Refined Professional CSS
st.markdown(
    """
    <style>
    .main { background-color: #f4f4f4; }
    
    /* Primary Action Button */
    .stButton>button { 
        background-color: #E02035; 
        color: white; 
        border-radius: 4px; 
        font-weight: 700;
        border: none;
    }

    /* The High-Contrast Intent Card */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 8px solid #E02035; /* Fanshawe Red Accent */
        padding: 25px;
        border-radius: 4px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    /* Forcing Deep Black/Charcoal for the Result Text */
    div[data-testid="stMetricValue"] > div {
        color: #1A1A1A !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }

    /* Forcing Dark Grey for the Label */
    div[data-testid="stMetricLabel"] > div {
        color: #444444 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("<h1 style='font-size: 70px; margin-top: -15px;'>🌐</h1>", unsafe_allow_html=True)

with col2:
    st.title("VantagePay Intent Engine")
    st.markdown("<span style='color: #E02035;'>**Enterprise NLP Microservice**</span>", unsafe_allow_html=True)

st.divider()

# Project Context
with st.expander("📖 About this Architecture"):
    st.write(
        """
        **VantagePay Intent Engine** is a high-precision classification system developed to automate 
        the triage of unstructured customer support queries. 
        
        The engine utilizes a **Linear Support Vector Machine (LinearSVC)** paired with a **10,000-feature 
        TF-IDF vectorizer**. This architecture was selected for its performance in high-dimensional 
        sparse vector spaces, enabling accurate disambiguation across 27 business intents.
        """
    )

with st.expander("🛠 How to Use"):
    st.write(
        """
        1. **Input:** Provide a raw customer utterance in the text area.
        2. **Process:** Execute the 'Analyze & Route' command to trigger the preprocessing pipeline.
        3. **Normalization:** The engine lemmatizes the input, isolating semantic roots for classification.
        4. **Output:** The predicted business intent and the automated routing destination are displayed.
        """
    )

# Input Section
st.subheader("Customer Inquiry Analysis")
user_input = st.text_area(
    "Enter raw message for real-time classification:",
    height=150,
    placeholder="e.g., I'm trying to track my refund from last Tuesday...",
)

# --- 3. INFERENCE ENGINE ---
if st.button("Analyze & Route", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.error("Submission requires valid text input.")
    else:
        with st.spinner("Processing semantics..."):
            clean_text = lemmatize_text(user_input)
            vectorized_text = tfidf.transform([clean_text])
            prediction_encoded = model.predict(vectorized_text)
            final_intent = encoder.inverse_transform(prediction_encoded)[0]

        # Results visualization
        st.success("Analysis Complete")

        # The Metric Box (Now styled with the new CSS)
        st.metric(
            label="Verified Business Intent",
            value=final_intent.replace("_", " ").upper(),
        )

        # Routing Info
        dept = final_intent.split("_")[-1].upper() if "_" in final_intent else "GENERAL SUPPORT"
        st.info(f"**Operational Logic:** Routing to **{dept}** division.")

# --- 4. AUTHORSHIP ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888888; font-size: 0.85em;'>
        Designed and Engineered by <strong>Moses Mudiaga Effeyotah</strong><br>
        School of Information Technology | Fanshawe College
    </div>
    """,
    unsafe_allow_html=True,
)
