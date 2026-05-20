import os
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(duration: int = 5) -> bytes:
    """
    Records audio from the microphone for a given duration.
    Returns raw audio bytes in WAV format.
    """
    print(f"Recording for {duration} seconds... Speak now.")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )
    sd.wait()
    print("Recording complete.")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return buffer.read()


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Sends audio bytes to Groq Whisper and returns transcribed text.
    """
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language="en",
        response_format="text"
    )

    if not isinstance(transcription, str):
        return ""

    return transcription.strip()


def listen_and_transcribe(duration: int = 5) -> str:
    """
    Full pipeline: record from mic, transcribe via Groq Whisper.
    Returns the transcribed string.
    """
    audio_bytes = record_audio(duration)
    text = transcribe_audio(audio_bytes)
    print(f"Transcribed: {text}")
    return text


if __name__ == "__main__":
    result = listen_and_transcribe(duration=5)
    print(f"Final output: {result}")