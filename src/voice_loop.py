from stt import listen_and_transcribe
from tts import speak
from agent import get_response


EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "thank you bye"]


def run_voice_loop():
    conversation_history = []

    speak("Hello, thank you for calling McDonald's crew support. My name is Max. How can I help you today?")

    while True:
        user_input = listen_and_transcribe(duration=6)

        if not user_input or len(user_input.strip()) < 3:
            speak("I did not catch that. Could you please repeat?")
            continue

        if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
            speak("Thank you for calling McDonald's crew support. Have a good day.")
            break

        conversation_history.append({"role": "user", "content": user_input})
        response = get_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

        interrupted = speak(response)

        if interrupted:
            speak("Go ahead, I am listening.")


if __name__ == "__main__":
    run_voice_loop()