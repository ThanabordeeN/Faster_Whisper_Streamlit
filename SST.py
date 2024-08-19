import streamlit as st
import speech_recognition as sr
import wave
import io
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

st.title("Speech-to-Text (Thai)")
model_options = ["small","tiny"]
model_size = st.selectbox("Select Model Size", model_options, index=0)

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
#model = WhisperModel(model_size, device="cpu", compute_type="int8")
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# Select device and compute type
device_options = ["cpu", "cuda"]
device = st.selectbox("Select Device", device_options, index=0)

compute_type_options = ["float16", "int8", "int8_float16"]
compute_type = st.selectbox("Select Compute Type", compute_type_options, index=0)

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
                try:
                    # Save audio as mp3
                    # Convert audio data to numpy array
                    audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                    # Create a wave object
                    wave_obj = wave.open(io.BytesIO(audio.get_wav_data()), 'rb')
                    # Get the frame rate and number of channels
                    frame_rate = wave_obj.getframerate()
                    channels = wave_obj.getnchannels()
                    # Create an AudioSegment object
                    audio_segment = AudioSegment(
                        audio_data.tobytes(),
                        frame_rate=frame_rate,
                        channels=channels,
                        sample_width=2,
                    )
                    # Save the audio as mp3
                    audio_segment.export("./temp/recorded_audio.mp3", format="mp3")

                    # or run on GPU with INT8
                    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                    # or run on CPU with INT8
                    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

                    segments, info = model.transcribe("./temp/recorded_audio.mp3", beam_size=5 ,language="en", condition_on_previous_text=False)

                    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

                    for segment in segments:
                        st.text("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    listening = False
                    st.write("Stopped listening.")

                except Exception as e:
                    st.write(f"{e}")

if stop:
    listening = False
    st.write("Stopped listening.")
