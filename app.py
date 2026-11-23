"""Stage 9: Streamlit GUI.  Run: streamlit run app.py"""
import json, os
import streamlit as st

st.set_page_config(page_title="EN→FR Translator", page_icon="🌐")
st.title("🌐 English → French Translator")

engine = st.sidebar.radio("Model", ["Custom Seq2Seq LSTM (trained from scratch)",
                                    "Pretrained MarianMT (Hugging Face)"])
if os.path.exists("model/metrics.json"):
    st.sidebar.subheader("Custom model metrics")
    for k, v in json.load(open("model/metrics.json")).items():
        st.sidebar.write(f"**{k}**: {v:.3f}")

@st.cache_resource
def load_custom():
    from translator import Translator
    return Translator()

@st.cache_resource
def load_marian():
    from transformers import pipeline
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

text = st.text_area("Enter English text", "How are you today?", height=120)
if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Translating..."):
                if engine.startswith("Custom"):
                    result = " ".join(load_custom().translate(l) for l in text.split("\n") if l.strip())
                else:
                    result = load_marian()(text, max_length=256)[0]["translation_text"]
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}. Did you run prepare_data.py and train.py first?")
st.caption("Custom model works best on short sentences (≤12 words).")
