import speech_recognition as sr
from faster_whisper import WhisperModel
import tempfile
import os

# Initialize the Whisper model
model_size = "tiny"  # Choose the model size that suits your needs
model = WhisperModel(model_size,device='cuda')

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

def transcribe_audio(audio_data):
    # Save the audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name

    # Transcribe the audio file using Faster Whisper
    segments, info = model.transcribe(temp_audio_file_path, beam_size=5 ,language='en' ,condition_on_previous_text=False)
    transcription = " ".join([segment.text for segment in segments])

    # Clean up the temporary file
    os.remove(temp_audio_file_path)

    return transcription

def main():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)

        while True:
            try:
                print("Listening...")
                # Capture audio from the microphone
                audio = recognizer.listen(source)
                print("Transcripting...")
                # Convert the audio to text
                text = transcribe_audio(audio.get_wav_data())
                
                # Print the transcription
                print("You said: " + text)
                
            except sr.WaitTimeoutError:
                print("Listening...")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Error; {0}".format(e))

if __name__ == "__main__":
    main()