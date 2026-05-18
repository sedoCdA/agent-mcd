from stt import listen_and_transcribe
from tts import speak
from agent import get_response


STOP_PHRASES = ["stop", "wait", "hold on", "pause", "let me", "actually"]
EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "thank you bye"]


def check_for_interrupt(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in STOP_PHRASES)


def run_voice_loop():
    conversation_history = []

    speak("Hello, thank you for calling McDonald's crew support. My name is Max. How can I help you today?")

    while True:
        user_input = listen_and_transcribe(duration=6)

        if not user_input:
            speak("I did not catch that. Could you please repeat your issue?")
            continue

        if check_for_interrupt(user_input):
            speak("Of course, take your time. I am listening.")
            continue

        if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
            speak("Thank you for calling McDonald's crew support. Have a good day.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        response = get_response(conversation_history)

        conversation_history.append({"role": "assistant", "content": response})

        speak(response)


if __name__ == "__main__":
    run_voice_loop()