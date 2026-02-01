# app.py
"""
🌐 Interactive Translator App using Groq + Hugging Face
Run locally: streamlit run app.py

Instructions:
1. Set your Groq API key:
   import os
   os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
2. Install dependencies:
   pip install streamlit transformers torch groq sentencepiece
"""

import os
import streamlit as st
from typing import List, Optional

# -------------------------------
# Groq API
# -------------------------------
from groq import Groq

# Hugging Face
from transformers import MarianMTModel, MarianTokenizer

# -------------------------------
# Groq Translator
# -------------------------------
class GroqTranslator:
    def __init__(self, api_key: Optional[str] = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model_name = "llama-3.3-70b-versatile"

    def translate(self, text: str, source_lang: str = "", target_lang: str = "en") -> str:
        prompt = f"Translate the following text from {source_lang or 'auto-detect'} to {target_lang}:\n{text}"
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq translation error: {e}"

# -------------------------------
# Hugging Face fallback
# -------------------------------
class HFTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return "\n".join(translated_text)

# -------------------------------
# Language options
# -------------------------------
LANGUAGES = [
    "Auto-detect", "English", "Spanish", "French", "German",
    "Chinese", "Urdu", "Arabic", "Japanese", "Russian"
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌐 Translator App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🌐 Translator App (Groq + HF)</h1>", unsafe_allow_html=True)

# Initialize session state for history and input
if "history" not in st.session_state:
    st.session_state.history: List[str] = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -------------------------------
# Layout
# -------------------------------
with st.container():
    st.markdown("<h3 style='color:#306998;'>📝 Enter Text</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background-color: #D0E8F2; padding:10px; border-radius:10px;'>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([1, 1])

    # Language dropdowns
    with col1:
        source_lang = st.selectbox("Source Language", LANGUAGES, index=0)
    with col2:
        target_lang = st.selectbox("Target Language", LANGUAGES, index=1 if "English" in LANGUAGES else 0)

    # Text input area
    user_input = st.text_area("Enter text to translate", height=150, value=st.session_state.input_text)

    # Clear input button
    if st.button("🧹 Clear Input"):
        user_input = ""
        st.session_state.input_text = ""

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Translation logic
# -------------------------------
if st.button("🚀 Translate"):
    if not user_input.strip():
        st.warning("⚠️ Please enter text to translate!")
    else:
        st.session_state.input_text = user_input  # preserve input
        groq_translator = GroqTranslator()
        translated_text = groq_translator.translate(user_input, source_lang, target_lang)

        # Fallback to Hugging Face
        if "error" in translated_text.lower():
            st.info("⚠️ Groq failed. Using Hugging Face fallback...")
            hf_translator = HFTranslator()
            translated_text = hf_translator.translate(user_input)

        st.markdown("<h3 style='color:#228B22;'>✅ Translation Complete!</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color: #DFF2BF; padding:10px; border-radius:10px;'>{translated_text}</div>",
            unsafe_allow_html=True
        )
        st.session_state.history.append(translated_text)

# -------------------------------
# History Section
# -------------------------------
if st.session_state.history:
    st.markdown("<h3 style='color:#E6B800;'>📜 Translation History</h3>", unsafe_allow_html=True)
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(
            f"<div style='background-color: #FFF8DC; padding:8px; border-radius:8px; margin-bottom:5px;'>"
            f"{idx}. {item}<br><small>Tip: Select and copy with Ctrl+C</small></div>",
            unsafe_allow_html=True
        )

# -------------------------------
# Clear history
# -------------------------------
if st.button("🗑 Clear History"):
    st.session_state.history.clear()
    st.success("🧹 History cleared!")

# -------------------------------
# Footer instructions
# -------------------------------
st.markdown("<i>💡 To copy translated text, select it and press Ctrl+C (Cmd+C on Mac).</i>", unsafe_allow_html=True)
