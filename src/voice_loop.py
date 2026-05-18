from stt import listen_and_transcribe
from tts import speak, configure_voice


STOP_PHRASES = ["stop", "wait", "hold on", "pause", "let me", "actually"]


def check_for_interrupt(text: str) -> bool:
    """
    Checks if the user wants to pause or correct themselves.
    """
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in STOP_PHRASES)


def run_voice_loop():
    configure_voice()
    speak("Hello, welcome to McDonald's crew support. I am here to help you. Please describe your issue.")

    while True:
        user_input = listen_and_transcribe(duration=6)

        if not user_input:
            speak("I did not catch that. Could you please repeat?")
            continue

        if check_for_interrupt(user_input):
            speak("Of course, take your time. I am listening.")
            continue

        if any(word in user_input.lower() for word in ["exit", "quit", "goodbye", "bye"]):
            speak("Thank you for calling. Goodbye.")
            break

        # Placeholder echo response before real agent logic
        speak(f"I heard you say: {user_input}. Let me process that.")


if __name__ == "__main__":
    run_voice_loop()