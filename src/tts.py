import asyncio
import edge_tts
import tempfile
import os
import sounddevice as sd
import soundfile as sf


VOICE = "en-US-AriaNeural"
RATE = "+0%"
VOLUME = "+0%"


async def _speak_async(text: str):
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, volume=VOLUME)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name

    await communicate.save(tmp_path)

    data, samplerate = sf.read(tmp_path)
    sd.play(data, samplerate)
    sd.wait()
    os.unlink(tmp_path)


def speak(text: str):
    """
    Converts text to speech using Microsoft Edge neural TTS.
    Plays audio through speakers and prints to console.
    """
    print(f"Agent: {text}")
    asyncio.run(_speak_async(text))


if __name__ == "__main__":
    speak("Hello, this is the McDonald's crew support assistant. How can I help you today?")
    speak("I understand your issue. Let me look into that for you.")
    speak("Your problem has been escalated to a human agent. Please hold.")