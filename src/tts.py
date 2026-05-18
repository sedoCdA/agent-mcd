import pyttsx3


engine = pyttsx3.init()


def configure_voice(rate: int = 165, volume: float = 1.0):
    """
    Configure speech rate and volume.
    rate: words per minute (default 165, calm and clear)
    volume: 0.0 to 1.0
    """
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    voices = engine.getProperty("voices")
    for voice in voices:
        if "english" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break


def speak(text: str):
    """
    Converts text to speech and plays it immediately.
    Blocks until speech is complete.
    """
    print(f"Agent: {text}")
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    configure_voice()
    speak("Hello, this is the McDonald's crew support assistant. How can I help you today?")
    speak("I understand your issue. Let me look into that for you.")
    speak("Your problem has been escalated to a human agent. Please hold.")