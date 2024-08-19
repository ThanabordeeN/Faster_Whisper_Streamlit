import streamlit as st
import speech_recognition as sr
import tempfile
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

st.title("Speech-to-Text (Thai)")
model_options = ["small", "tiny"]
model_size = st.selectbox("Select Model Size", model_options, index=0)

# Select device and compute type
device_options = ["cpu", "cuda"]
device = st.selectbox("Select Device", device_options, index=0)

compute_type_options = ["int8", "float16", "int8_float16"]
compute_type = st.selectbox("Select Compute Type", compute_type_options, index=0)
language_option = ['th', 'en']
language = st.selectbox("Select Language", language_option, index=0)


# Initialize Whisper model
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# Initialize Speech Recognition
recog = sr.Recognizer()

# Select microphone
mic = sr.Microphone(1)

# Start listening loop
listening = False
start = st.button("Start Listening")
stop = st.button("Stop Listening")

if start:
    listening = True
    while listening:
        with mic as source:
            while listening:
                audio = recog.listen(source)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                        temp_audio_file.write(audio.get_wav_data())
                        temp_audio_file_path = temp_audio_file.name
                    # Get the frame rate and number of channels

                # Transcribe the audio using Faster Whisper
                segments, info = model.transcribe(temp_audio_file_path, beam_size=5, language=language, condition_on_previous_text=False)

                print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

                for segment in segments:
                    st.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                
                listening = False
                st.write("Stopped listening.")

                

if stop:
    listening = False
    st.write("Stopped listening.")