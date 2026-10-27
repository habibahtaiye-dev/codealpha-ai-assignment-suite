import os
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Assignment Suite", page_icon="🤖", layout="wide")
st.title("🤖 AI Assignment Suite")
st.caption("Language Translation • FAQ Chatbot • AI Music Demo • Object Detection guide")

# FAQ data can be expanded for your chosen product/topic.
FAQS = [
    ("What is this project?", "This is a student demo combining a translation tool, FAQ chatbot, music-generation example, and object-detection example."),
    ("How do I use the translation tool?", "Enter text, choose the source and target languages, then click Translate. An internet connection and a working translation API are required."),
    ("Is my information stored?", "This demo does not intentionally save your text to a database. Text sent to a translation API is processed by that service, so avoid entering sensitive information."),
    ("How do I run object detection?", "Install the dependencies and run `python object_detection.py`. Allow webcam permissions, then press q to quit."),
    ("How does the FAQ chatbot work?", "It converts FAQ questions into TF-IDF vectors and ranks them against the user's question using cosine similarity."),
    ("Does the music model create professional music?", "No. The included LSTM is a small educational demonstration trained on synthetic note patterns, not a production-quality music model."),
    ("How do I submit this project?", "Upload the project files to a GitHub repository and submit the repository URL in your course portal. See README.md for the steps."),
]
LANGUAGES = {
    "English": "en", "French": "fr", "Spanish": "es", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Arabic": "ar", "Chinese": "zh",
    "Japanese": "ja", "Korean": "ko", "Hindi": "hi", "Russian": "ru",
    "Dutch": "nl", "Turkish": "tr", "Swahili": "sw"
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Translation", "💬 FAQ Chatbot", "🎼 Music Demo", "📷 Object Detection"
])

with tab1:
    st.subheader("Language Translation Tool")
    st.write("Translate text through a LibreTranslate-compatible endpoint.")
    source_label = st.selectbox("Source language", ["Auto-detect"] + list(LANGUAGES.keys()), index=0)
    target_label = st.selectbox("Target language", list(LANGUAGES.keys()), index=1)
    source_text = st.text_area("Text to translate", placeholder="Type or paste your text here…", height=160)
    if st.button("Translate", type="primary"):
        if not source_text.strip():
            st.warning("Please enter some text first.")
        elif source_label == target_label:
            st.info("Source and target languages are the same.")
        else:
            endpoint = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com/translate")
            payload = {
                "q": source_text,
                "source": "auto" if source_label == "Auto-detect" else LANGUAGES[source_label],
                "target": LANGUAGES[target_label],
                "format": "text",
            }
            api_key = os.getenv("LIBRETRANSLATE_API_KEY")
            if api_key:
                payload["api_key"] = api_key
            try:
                with st.spinner("Translating…"):
                    response = requests.post(endpoint, data=payload, timeout=30)
                response.raise_for_status()
                translated = response.json().get("translatedText")
                if translated:
                    st.markdown("#### Translated text")
                    st.text_area("Result", value=translated, height=160)
                    st.download_button("Download translation", translated, file_name="translation.txt")
                else:
                    st.error("The API response did not include translatedText.")
            except requests.RequestException as exc:
                st.error(f"Translation request failed: {exc}")
                st.info("A public endpoint may be unavailable or rate-limited. Configure LIBRETRANSLATE_URL and, if needed, LIBRETRANSLATE_API_KEY.")
            except (ValueError, KeyError):
                st.error("Could not read the API response. Check that the endpoint is a LibreTranslate-compatible API.")

with tab2:
    st.subheader("FAQ Chatbot")
    st.write("Ask a question. The chatbot ranks the closest FAQ using TF-IDF and cosine similarity.")
    questions = [item[0] for item in FAQS]
    answers = [item[1] for item in FAQS]
    user_question = st.text_input("Your question", placeholder="e.g. How do I submit this project?")
    threshold = st.slider("Minimum similarity to accept a match", 0.0, 1.0, 0.15, 0.05)
    if st.button("Ask chatbot"):
        if not user_question.strip():
            st.warning("Type a question to continue.")
        else:
            corpus = questions + [user_question]
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
            scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            best_idx = int(scores.argmax())
            st.markdown("#### Chatbot response")
            if scores[best_idx] >= threshold:
                st.success(answers[best_idx])
                st.caption(f"Matched FAQ: {questions[best_idx]} (similarity {scores[best_idx]:.2f})")
            else:
                st.info("I couldn't find a close match. Try asking about translation, the FAQ chatbot, music demo, object detection, privacy, or submission.")

with tab3:
    st.subheader("AI Music Generation")
    st.write("The separate `music_generator.py` file demonstrates training a small LSTM on synthetic note sequences.")
    st.warning("This is an educational prototype, not a trained professional music generator.")
    st.code("python music_generator.py", language="bash")
    st.write("If successful, it writes `generated_notes.mid`. Open the MIDI file in a MIDI player or digital audio workstation.")

with tab4:
    st.subheader("Object Detection and Tracking")
    st.write("The separate `object_detection.py` script uses a pretrained YOLO model and OpenCV webcam frames.")
    st.code("python object_detection.py", language="bash")
    st.write("Bounding boxes and class labels are shown in a live window. The included tracker assigns simple centroid-based IDs; it is a learning demo, not a robust Deep SORT tracker.")
