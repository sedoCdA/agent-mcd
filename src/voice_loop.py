import time
from stt import listen_and_transcribe
from tts import speak
from agent import get_response, extract_call_metadata
from logger import log_call
from escalation import EscalationManager

EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "thank you bye"]
MAX_ATTEMPTS = 3
DEFAULT_PRIORITY = "P2"


def get_caller_info() -> tuple:
    speak("Before we begin, could you please tell me your name?")
    name = listen_and_transcribe(duration=5)
    if not name or len(name.strip()) < 2:
        name = "Unknown"

    speak(f"Thank you {name}. And what is your store ID?")
    store_id = listen_and_transcribe(duration=5)
    if not store_id or len(store_id.strip()) < 2:
        store_id = "Unknown"

    speak(f"Got it. Store {store_id}. How can I help you today?")
    return name.strip(), store_id.strip()


def detect_priority_early(user_input: str) -> str:
    """
    Does a quick keyword scan to set initial priority
    before the full LLM metadata extraction runs.
    """
    text = user_input.lower()
    p1_keywords = ["completely down", "not working at all", "store down",
                   "no orders", "payment not working", "kiosk blank", "server off"]
    p2_keywords = ["not showing", "not syncing", "not updating",
                   "access denied", "not processing", "headset"]

    if any(kw in text for kw in p1_keywords):
        return "P1"
    if any(kw in text for kw in p2_keywords):
        return "P2"
    return DEFAULT_PRIORITY


def finalize_and_log(
    caller_name: str,
    store_id: str,
    conversation_history: list,
    escalation_manager: EscalationManager,
    resolution_status: str,
    attempts: int,
    start_time: float
):
    metadata = extract_call_metadata(conversation_history)
    duration = time.time() - start_time

    log_call(
        caller_name=caller_name,
        store_id=store_id,
        issue_description=metadata.get("issue_description", "N/A"),
        priority=escalation_manager.priority,
        resolution_status=resolution_status,
        escalated_to_human=escalation_manager.escalated,
        attempts=attempts,
        duration_seconds=duration
    )

    summary = escalation_manager.summary()
    print(f"Call summary: {summary}")


def run_voice_loop():
    conversation_history = []
    attempts = 0
    start_time = time.time()
    escalation_manager = None

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

            if escalation_manager:
                finalize_and_log(
                    caller_name, store_id, conversation_history,
                    escalation_manager, resolution, attempts, start_time
                )

            speak("Thank you for calling McDonald's crew support. Have a good day.")
            break

        conversation_history.append({"role": "user", "content": user_input})
        attempts += 1

        if escalation_manager is None:
            initial_priority = detect_priority_early(user_input)
            escalation_manager = EscalationManager(initial_priority)
            speak(f"I understand. This appears to be a {initial_priority} priority issue. Let me help you resolve it within {escalation_manager.sla_label}.")

        sla_warning = escalation_manager.get_sla_warning_message()
        if sla_warning:
            speak(sla_warning)

        should_escalate, reason = escalation_manager.should_escalate(attempts, MAX_ATTEMPTS)

        if should_escalate:
            escalation_manager.escalate(reason)
            escalation_message = escalation_manager.get_escalation_message(reason)
            speak(escalation_message)

            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Unresolved", attempts, start_time
            )
            break

        response = get_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

        interrupted = speak(response)
        if interrupted:
            speak("Go ahead, I am listening.")


if __name__ == "__main__":
    run_voice_loop()