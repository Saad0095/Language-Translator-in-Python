# app.py
"""
🌐 Advanced Translator App using Groq + Hugging Face
Run locally: streamlit run app.py

Instructions:
1. Set your Groq API key:
   import os
   os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
2. Install dependencies:
   pip install streamlit transformers torch groq sentencepiece

Features:
- Groq LLM as primary translator with HF fallback
- Beautiful color-coded UI sections
- History tracking and management
- Dropdown language selection
- Full error handling
"""

import os
import streamlit as st
from typing import List, Optional, Dict
from datetime import datetime

# ============================================
# IMPORTS: Groq & Hugging Face
# ============================================
from groq import Groq
from transformers import MarianMTModel, MarianTokenizer

# ============================================
# TRANSLATOR CLASSES
# ============================================

class GroqTranslator:
    """Groq-based translator using LLaMA 3.3-70B model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq translator with API key.
        
        Args:
            api_key: Groq API key (defaults to env variable)
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"

    def translate(self, text: str, source_lang: str = "Auto-detect", target_lang: str = "English") -> str:
        """
        Translate text using Groq LLaMA model.
        
        Args:
            text: Text to translate
            source_lang: Source language name
            target_lang: Target language name
            
        Returns:
            Translated text or error message
        """
        source_display = "auto-detect" if source_lang == "Auto-detect" else source_lang
        prompt = (
            f"Translate the following text from {source_display} to {target_lang}. "
            f"Return only the translated text, nothing else:\n\n{text}"
        )
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Groq Error: {str(e)}")


class HFTranslator:
    """Hugging Face MarianMT translator as fallback."""
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        """
        Initialize HF translator with model.
        
        Args:
            model_name: HF model identifier (Helsinki-NLP opus-mt models)
        """
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model_name = model_name
        except Exception as e:
            raise Exception(f"HF Model Load Error: {str(e)}")

    def translate(self, text: str) -> str:
        """
        Translate text using MarianMT model.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text or error message
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            translated_tokens = self.model.generate(**inputs)
            result = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            return "\n".join(result)
        except Exception as e:
            raise Exception(f"HF Translation Error: {str(e)}")


# ============================================
# LANGUAGE MAPPING
# ============================================

LANGUAGES = {
    "Auto-detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Urdu": "ur",
    "Arabic": "ar",
    "Japanese": "ja",
    "Russian": "ru",
}

LANGUAGE_NAMES = list(LANGUAGES.keys())

# ============================================
# STREAMLIT PAGE CONFIG & STYLING
# ============================================

st.set_page_config(
    page_title="🌐 Advanced Translator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
CUSTOM_CSS = """
<style>
    /* Input section styling (light blue) */
    .input-section {
        background-color: #D0E8F2;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 20px;
    }
    
    /* Output section styling (light green) */
    .output-section {
        background-color: #D4EDDA;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 20px;
    }
    
    /* History section styling (light yellow) */
    .history-section {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 10px;
    }
    
    /* Error message styling (light red) */
    .error-message {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #f5c6cb;
        margin: 10px 0;
        font-weight: 500;
    }
    
    /* Success message styling */
    .success-message {
        background-color: #D4EDDA;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #c3e6cb;
        margin: 10px 0;
    }
    
    /* History item styling */
    .history-item {
        background-color: #FFFACD;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 14px;
        border: 1px solid #FFE4B5;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "last_translation" not in st.session_state:
    st.session_state.last_translation = ""

if "source_lang_idx" not in st.session_state:
    st.session_state.source_lang_idx = 0  # Auto-detect

if "target_lang_idx" not in st.session_state:
    st.session_state.target_lang_idx = 1  # English

# ============================================
# HEADER
# ============================================

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4; font-size: 2.5em; margin-bottom: 10px;'>"
    "🌐 Translator</h1>",
    unsafe_allow_html=True
)

st.divider()

# ============================================
# INPUT SECTION
# ============================================

with st.container():
    st.markdown(
        "<div class='input-section'>"
        "<h2 style='color: #0055CC; margin-top: 0;'>📝 Input & Language Selection</h2>",
        unsafe_allow_html=True
    )
    
    # Language selection columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        source_lang_idx = st.selectbox(
            "📤 Source Language",
            range(len(LANGUAGE_NAMES)),
            index=st.session_state.source_lang_idx,
            format_func=lambda x: LANGUAGE_NAMES[x]
        )
        st.session_state.source_lang_idx = source_lang_idx
    
    with col2:
        target_lang_idx = st.selectbox(
            "📥 Target Language",
            range(len(LANGUAGE_NAMES)),
            index=st.session_state.target_lang_idx,
            format_func=lambda x: LANGUAGE_NAMES[x]
        )
        st.session_state.target_lang_idx = target_lang_idx
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
        if st.button("🔄 Swap Languages", use_container_width=True):
            # Swap the indices
            st.session_state.source_lang_idx, st.session_state.target_lang_idx = (
                st.session_state.target_lang_idx,
                st.session_state.source_lang_idx
            )
            st.rerun()
    
    # Get language names from indices
    source_lang = LANGUAGE_NAMES[st.session_state.source_lang_idx]
    target_lang = LANGUAGE_NAMES[st.session_state.target_lang_idx]
    
    # Text input area
    user_input = st.text_area(
        "💬 Enter text to translate",
        height=150,
        placeholder="Type or paste your text here...",
        value=st.session_state.input_text,
        label_visibility="collapsed"
    )
    
    # Update session state with current input
    st.session_state.input_text = user_input
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# ACTION BUTTONS
# ============================================

button_col1, button_col2, button_col3 = st.columns([1, 1, 2])

with button_col1:
    translate_btn = st.button(
        "🚀 Translate",
        use_container_width=True,
        type="primary"
    )

with button_col2:
    clear_input_btn = st.button(
        "🧹 Clear Input",
        use_container_width=True
    )

# Handle Clear Input
if clear_input_btn:
    st.session_state.input_text = ""
    st.session_state.last_translation = ""
    st.rerun()

# ============================================
# TRANSLATION LOGIC
# ============================================

if translate_btn:
    if not user_input.strip():
        st.markdown(
            "<div class='error-message'>⚠️ Please enter some text first.</div>",
            unsafe_allow_html=True
        )
    else:
        with st.spinner("⏳ Translating..."):
            try:
                # Try Groq first
                try:
                    groq_translator = GroqTranslator()
                    translated_text = groq_translator.translate(user_input, source_lang, target_lang)
                    provider = "Fast Mode"
                except Exception as groq_error:
                    st.info("⏱️ Using alternative translator...")
                    provider = "Alternative Mode"
                    
                    hf_translator = HFTranslator()
                    translated_text = hf_translator.translate(user_input)
                
                # Store translation in session state
                st.session_state.last_translation = translated_text
                
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source_lang,
                    "target": target_lang,
                    "original": user_input[:100] + ("..." if len(user_input) > 100 else ""),
                    "translation": translated_text[:100] + ("..." if len(translated_text) > 100 else ""),
                    "full_translation": translated_text,
                    "provider": provider
                }
                st.session_state.history.append(history_entry)
                
            except Exception as e:
                st.markdown(
                    "<div class='error-message'>❌ Something went wrong. Please try again.</div>",
                    unsafe_allow_html=True
                )

# ============================================
# OUTPUT SECTION
# ============================================

if st.session_state.last_translation:
    st.markdown("")  # Spacing
    
    with st.container():
        st.markdown(
            "<div class='output-section'>"
            "<h2 style='color: #155724; margin-top: 0;'>✅ Translation Result</h2>",
            unsafe_allow_html=True
        )
        
        # Display translation
        st.code(st.session_state.last_translation, language="text")
        
        # Copy instruction
        st.markdown(
            "<p style='color: #666; font-size: 0.9em;'>"
            "📋 Click the copy button in the corner or select and copy</p>",
            unsafe_allow_html=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# HISTORY SECTION
# ============================================

st.divider()

if st.session_state.history:
    st.markdown(
        "<h2 style='color: #856404;'>📜 Translation History</h2>",
        unsafe_allow_html=True
    )
    
    # History controls
    hist_col1, hist_col2 = st.columns([3, 1])
    
    with hist_col2:
        clear_history_btn = st.button(
            "🗑️ Clear History",
            use_container_width=True
        )
    
    if clear_history_btn:
        st.session_state.history.clear()
        st.session_state.last_translation = ""
        st.success("✓ History cleared")
        st.rerun()
    
    # Display history items
    displayed_count = 0
    for idx, entry in enumerate(reversed(st.session_state.history), start=1):
        
        displayed_count += 1
        
        with st.expander(
            f"#{idx} | {entry['source']} → {entry['target']} | {entry['timestamp']}",
            expanded=False
        ):
            st.markdown(
                f"<div class='history-item'>"
                f"<b>Original:</b> {entry['original']}<br>"
                f"<b>Result:</b> {entry['translation']}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Display full translation in code block
            st.code(entry['full_translation'], language="text")
    
    if displayed_count == 0:
        st.info("ℹ️ No translations yet.")

else:
    st.info("ℹ️ No translation history yet. Start by entering text and clicking 'Translate'.")

# ============================================
# FOOTER
# ============================================

st.divider()

st.markdown(
    "<footer style='text-align: center; color: #999; font-size: 0.9em; margin-top: 20px;'>"
    "🌐 Translator | Supports 9 languages"
    "</footer>",
    unsafe_allow_html=True
)
