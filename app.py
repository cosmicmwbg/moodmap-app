
import streamlit as st
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MoodMap | Emotion AI",
    page_icon="🧠",
    layout="wide"
)

MODEL_ID = "sbiii555494/moodmap-distilbert-v6" # <-- Updated to V6
MAX_LENGTH = 128

EMOTION_ICONS = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐" # <-- Added neutral icon
}

# ==========================================
# 2. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #101827;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1050px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 16px;
    }

    .result-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }

    .emotion {
        font-size: 42px;
        font-weight: bold;
        color: #60A5FA;
    }

    [data-testid="stMetric"] {
        background: #1E293B;
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD FINE-TUNED MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_model():

    torch.set_num_threads(2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    model.eval()

    return tokenizer, model


# ==========================================
# 4. EMOTION PREDICTION
# ==========================================

def predict_emotion(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_id = torch.argmax(probabilities).item()

    labels = {
        int(k): v
        for k, v in model.config.id2label.items()
    }

    prediction = labels[predicted_id]

    results = {
        labels[i]: probabilities[i].item()
        for i in range(len(labels))
    }

    return prediction, results


# ==========================================
# 5. APPLICATION INTERFACE
# ==========================================

st.title("🧠 MoodMap")

st.markdown(
    '<p class="subtitle">'
    'Understand the emotional tone of your text '
    'using a fine-tuned DistilBERT Transformer.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Analyse your text")

EXAMPLES = {
    "Select an example": "",
    "Joy": "I finally achieved my dream after years of hard work!",
    "Sadness": "I feel completely abandoned and nobody understands me.",
    "Love": "I love spending time with my family.",
    "Anger": "I am absolutely furious about what happened.",
    "Fear": "I am terrified of what tomorrow might bring.",
    "Surprise": "I cannot believe this actually happened!"
}

selected_example = st.selectbox(
    "Try a sample sentence",
    list(EXAMPLES.keys())
)

if selected_example != "Select an example":
    st.session_state["user_text"] = EXAMPLES[selected_example]

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

user_text = st.text_area(
    "Enter your text below:",
    key="user_text",
    height=140,
    placeholder="Write something here..."
)

analyze = st.button(
    "🔍 Analyse Emotion",
    type="primary",
    use_container_width=True
)


# ==========================================
# 6. DISPLAY PREDICTIONS
# ==========================================

if analyze:

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        try:

            with st.spinner(
                "Loading MoodMap and analysing your text..."
            ):

                tokenizer, model = load_model()

                prediction, probabilities = predict_emotion(
                    user_text,
                    tokenizer,
                    model
                )

            st.divider()

            st.subheader("Analysis Results")

            icon = EMOTION_ICONS.get(
                prediction.lower(), "🧠"
            )

            confidence = probabilities[prediction] * 100

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Emotion",
                    f"{icon} {prediction.title()}"
                )

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.subheader("Emotion Probability Distribution")

            sorted_results = sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for emotion, probability in sorted_results:

                emotion_icon = EMOTION_ICONS.get(
                    emotion.lower(), "🧠"
                )

                st.write(
                    f"{emotion_icon} **{emotion.title()}**"
                    f" | {probability * 100:.2f}%"
                )

                st.progress(probability)

            st.divider()

            st.caption(
                "MoodMap uses a fine-tuned DistilBERT model "
                "trained on the DAIR.AI Emotion dataset. "
                "Predictions represent text-classification "
                "probabilities, not a diagnosis or verified "
                "assessment of a person's actual emotional state."
            )

        except Exception as error:

            st.error(
                "Something went wrong while analysing the text."
            )

            st.exception(error)


# ==========================================
# 7. SIDEBAR INFORMATION
# ==========================================

with st.sidebar:

    st.title("About MoodMap")

    st.write(
        "MoodMap is a Transformer-based "
        "emotion classification application."
    )

    st.divider()

    st.subheader("Model Information")
    
    st.write("**Architecture:** DistilBERT")
    
    st.write("**Model Version:** V5 (Includes Negation Handling)") # <-- Updated
    
    st.write("**Classes:** 7 emotions (added 'neutral')")          # <-- Updated
    
    st.write("**Training Samples:** 16,600+")                      # <-- Updated

    st.write("**Testing Samples:** 2,000")

    st.divider()

    st.subheader("Test Performance")

    st.metric("Test Accuracy", "92.25%")

    st.metric("Macro F1-Score", "87.35%")

    st.divider()

    st.caption(
        "Developed as an MDM LLM course project."
    )
