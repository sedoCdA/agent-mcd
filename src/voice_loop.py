import time
from stt import listen_and_transcribe
from tts import speak
from agent import get_response, extract_call_metadata
from logger import log_call

EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "thank you bye"]
MAX_ATTEMPTS = 3


def get_caller_info() -> tuple:
    """
    Collects caller name and store ID at the start of the call.
    """
    speak("Before we begin, could you please tell me your name?")
    name = listen_and_transcribe(duration=5)
    if not name:
        name = "Unknown"

    speak(f"Thank you {name}. And what is your store ID?")
    store_id = listen_and_transcribe(duration=5)
    if not store_id:
        store_id = "Unknown"

    speak(f"Got it. Store {store_id}. How can I help you today?")
    return name.strip(), store_id.strip()


def run_voice_loop():
    conversation_history = []
    attempts = 0
    escalated = False
    start_time = time.time()

    speak("Hello, thank you for calling McDonald's crew support. My name is Max.")

    caller_name, store_id = get_caller_info()

    while True:
        user_input = listen_and_transcribe(duration=6)

        if not user_input or len(user_input.strip()) < 3:
            speak("I did not catch that. Could you please repeat?")
            continue

        if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
            speak("Before you go, was your issue resolved today?")
            resolution_input = listen_and_transcribe(duration=4)
            resolution = "Resolved" if resolution_input and "yes" in resolution_input.lower() else "Unresolved"

            metadata = extract_call_metadata(conversation_history)
            duration = time.time() - start_time

            log_call(
                caller_name=caller_name,
                store_id=store_id,
                issue_description=metadata.get("issue_description", "N/A"),
                priority=metadata.get("priority", "P3"),
                resolution_status=resolution,
                escalated_to_human=escalated,
                attempts=attempts,
                duration_seconds=duration
            )

            speak("Thank you for calling McDonald's crew support. Have a good day.")
            break

        conversation_history.append({"role": "user", "content": user_input})
        attempts += 1

        if attempts >= MAX_ATTEMPTS and not escalated:
            speak("I have tried my best to resolve this issue. I am now escalating this to a human agent who will contact you shortly. Please stay available.")
            escalated = True

            metadata = extract_call_metadata(conversation_history)
            duration = time.time() - start_time

            log_call(
                caller_name=caller_name,
                store_id=store_id,
                issue_description=metadata.get("issue_description", "N/A"),
                priority=metadata.get("priority", "P1"),
                resolution_status="Unresolved",
                escalated_to_human=True,
                attempts=attempts,
                duration_seconds=duration
            )
            break

        response = get_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

        interrupted = speak(response)

        if interrupted:
            speak("Go ahead, I am listening.")


if __name__ == "__main__":
    run_voice_loop()