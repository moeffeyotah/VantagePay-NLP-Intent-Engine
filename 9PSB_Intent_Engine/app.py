import streamlit as st
import joblib
import spacy
import os


# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load serialized AI assets
    model = joblib.load(os.path.join(current_dir, "fanshawe_intent_svc_model.pkl"))
    tfidf = joblib.load(os.path.join(current_dir, "fanshawe_tfidf_vectorizer.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "fanshawe_label_encoder.pkl"))

    # Load NLP engine
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except:
        # Fallback if model isn't linked correctly in the environment
        import en_core_web_sm

        nlp = en_core_web_sm.load(disable=["parser", "ner"])

    return model, tfidf, encoder, nlp


model, tfidf, encoder, nlp = load_assets()


def lemmatize_text(text):
    """
    REQUIRED: Processes raw customer input into a clean format.
    Without this, the model cannot understand the user's message.
    """
    doc = nlp(str(text).lower())
    return " ".join(
        [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    )


# --- THE VANTAGEPAY USER INTERFACE ---
st.set_page_config(page_title="VantagePay Intent Engine", page_icon="🌐")

# Header Section
col1, col2 = st.columns([1, 5])
with col1:
    # A high-tech globe emoji works perfectly as a minimalist logo
    st.markdown(
        "<h1 style='font-size: 70px; margin-top: -15px;'>🌐</h1>",
        unsafe_allow_html=True,
    )

with col2:
    st.title("VantagePay Automated Triage")
    st.markdown(
        "<span style='color: #64FFDA;'>**Enterprise NLP Microservice**</span> | *Architect: Moses Effeyotah*",
        unsafe_allow_html=True,
    )

st.divider()

# Input Area
st.subheader("Simulate Transactional Inquiry")
user_input = st.text_area(
    "Enter customer message for real-time semantic analysis:",
    height=150,
    placeholder="e.g., My transfer to the UK failed but the funds were debited. Help!",
)

# --- THE INFERENCE ENGINE ---
if st.button("Analyze & Route", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.error("Input required for analysis.")
    else:
        with st.spinner("Executing Neural-Semantic Mapping..."):
            clean_text = lemmatize_text(user_input)
            vectorized_text = tfidf.transform([clean_text])
            prediction_encoded = model.predict(vectorized_text)
            final_intent = encoder.inverse_transform(prediction_encoded)[0]

        # Display Results
        st.success("Triage Complete: Stakeholder Identified")

        # Metric showing the detected intent
        st.metric(label="Detected Intent", value=final_intent.replace("_", " ").upper())

        # Final routing info box
        dept = final_intent.split("_")[-1].upper()
        st.info(
            f"**Action Required:** Automatically routing to the **{dept}** department."
        )
