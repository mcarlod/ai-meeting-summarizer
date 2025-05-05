import streamlit as st
import tempfile
import whisper
import openai
from io import StringIO

st.set_page_config(page_title="AI Meeting Summarizer", layout="centered")

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #333;
    }
    .stTextArea textarea {
        background-color: #f9f9f9;
        color: #333;
        font-family: 'Courier New', monospace;
    }
    .stSpinner > div {
        color: #6c6c6c;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎙️ AI Meeting & Podcast Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload your audio file and get a clean, concise summary in seconds.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def summarize_text(transcript):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful meeting summarizer."},
            {"role": "user", "content": f"Summarize this transcript:\n\n{transcript}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

st.markdown("### 📤 Upload Your Audio File")
uploaded_file = st.file_uploader("", type=["mp3", "wav", "m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    with st.spinner("🧠 Transcribing your file..."):
        transcript = transcribe_audio(file_path)

    with st.spinner("✍️ Generating your summary..."):
        summary = summarize_text(transcript)

    st.markdown("## 📝 Summary")
    st.success(summary)

    with st.expander("📄 Full Transcript"):
        st.text_area("", transcript, height=300)

    summary_file = StringIO(summary)
    st.download_button("📥 Download Summary", summary_file, file_name="summary.txt")
