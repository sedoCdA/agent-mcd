import asyncio
import edge_tts
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading


VOICE = "en-US-AriaNeural"
ENERGY_THRESHOLD = 1500


async def _generate_audio(text: str) -> tuple:
    communicate = edge_tts.Communicate(text, VOICE)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name
    await communicate.save(tmp_path)
    data, samplerate = sf.read(tmp_path)
    os.unlink(tmp_path)
    return data, samplerate


def speak(text: str) -> bool:
    """
    Speaks text aloud while monitoring the microphone.
    Stops immediately if the user starts speaking.
    Returns True if interrupted, False if completed normally.
    """
    print(f"Agent: {text}")

    data, samplerate = asyncio.run(_generate_audio(text))
    interrupted = threading.Event()

    def mic_callback(indata, frames, time, status):
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
        if rms > ENERGY_THRESHOLD:
            interrupted.set()
            sd.stop()

    mic_stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        callback=mic_callback,
        blocksize=int(16000 * 0.2)
    )

    with mic_stream:
        sd.play(data, samplerate)
        sd.wait()

    return interrupted.is_set()


if __name__ == "__main__":
    was_interrupted = speak("Hello, this is McDonald's crew support. My name is Max. Please describe your issue in detail so I can help you resolve it as quickly as possible.")
    if was_interrupted:
        print("User interrupted the bot.")
    else:
        print("Bot finished speaking normally.")