import streamlit as st
import joblib
import spacy
import os


# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load serialized AI assets
    # Ensure these filenames match your repo exactly
    model = joblib.load(os.path.join(current_dir, "fanshawe_intent_svc_model.pkl"))
    tfidf = joblib.load(os.path.join(current_dir, "fanshawe_tfidf_vectorizer.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "fanshawe_label_encoder.pkl"))

    # Load NLP engine
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


# --- THE VANTAGEPAY USER INTERFACE ---
st.set_page_config(
    page_title="VantagePay Intent Engine", page_icon="🌐", layout="centered"
)

# Custom CSS for the "Fanshawe Light" Aesthetic
st.markdown(
    """
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button { background-color: #E02035; color: white; border-radius: 5px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(
        "<h1 style='font-size: 70px; margin-top: -15px;'>🌐</h1>",
        unsafe_allow_html=True,
    )

with col2:
    st.title("VantagePay Intent Engine")
    st.markdown(
        "<span style='color: #E02035;'>**Enterprise NLP Microservice**</span>",
        unsafe_allow_html=True,
    )

st.divider()

# --- NEW SECTIONS: ABOUT & HOW TO USE ---
with st.expander("📖 About this Architecture"):
    st.write(
        """
        **VantagePay Intent Engine** is a high-precision NLP classifier designed to bridge the gap between 
        unstructured customer communication and automated operational efficiency. 
        
        Using a **Linear Support Vector Machine (LinearSVC)** and a **10,000-dimensional TF-IDF vectorizer**, 
        the engine identifies semantic patterns across 27 distinct business intents. This allows for 
        instant, zero-latency triage of customer support tickets, reducing SLA breaches and operational overhead.
    """
    )

with st.expander("🛠 How to Use"):
    st.write(
        """
        1. **Input:** Type or paste a raw customer query into the text area below.
        2. **Analyze:** Click the 'Analyze & Route' button to trigger the neural-semantic pipeline.
        3. **Lemmatization:** The engine will strip 'noise' (stopwords) and reduce words to their semantic roots.
        4. **Triage:** The detected intent and the designated department for routing will be displayed instantly.
    """
    )

# Input Area
st.subheader("Customer Query Analysis")
user_input = st.text_area(
    "Enter raw message for real-time classification:",
    height=150,
    placeholder="e.g., Where is the refund for my cancelled order? I've been waiting for three days.",
)

# --- THE INFERENCE ENGINE ---
if st.button("Analyze & Route", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.error("Input required for analysis.")
    else:
        with st.spinner("Executing Semantic Mapping..."):
            clean_text = lemmatize_text(user_input)
            vectorized_text = tfidf.transform([clean_text])
            prediction_encoded = model.predict(vectorized_text)
            final_intent = encoder.inverse_transform(prediction_encoded)[0]

        # Display Results
        st.success("Triage Successful")

        # Detected Intent Metric
        st.metric(
            label="Predicted Business Intent",
            value=final_intent.replace("_", " ").upper(),
        )

        # Logic to suggest department
        dept = (
            final_intent.split("_")[-1].upper()
            if "_" in final_intent
            else "GENERAL SUPPORT"
        )
        st.info(
            f"**Operational Action:** Automatically routing to the **{dept}** department."
        )

# Footer / Authorship
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888888; font-size: 0.8em;'>
        Designed and Engineered by <strong>Moses Mudiaga Effeyotah</strong><br>
        School of Information Technology | Fanshawe College
    </div>
    """,
    unsafe_allow_html=True,
)
