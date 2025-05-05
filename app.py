import streamlit as st
import tempfile
import whisper
import openai
import os

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_model():
    return whisper.load_model("base")
  
model = load_model()

def transcribe_audio(file_path):
  result = model.transcribe(file_path)
  return result['text']

def summarize_text(transcript):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful meeting summarizer."},
            {"role": "user", "content": f"Summarize this transcript:\n\n{transcript}"}
        ]
    )
    return response['choices'][0]['message']['content']

st.title("🎙️ AI Meeting / Podcast Summarizer")

uploaded_file = st.file_uploader("Upload your audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.info("File uploaded. Processing...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.text("Transcribing audio...")
    transcript = transcribe_audio(file_path)

    st.text("Generating summary...")
    summary = summarize_text(transcript)

    # Show summary and full transcript
    st.subheader("📝 Summary")
    st.write(summary)

    st.subheader("📄 Full Transcript")
    with st.expander("Click to expand"):
        st.write(transcript)