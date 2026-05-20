import re
import time

from stt import listen_and_transcribe
from tts import speak
from agent import get_response, extract_call_metadata, client, MODEL
from logger import log_call
from escalation import EscalationManager
from rag import retrieve_solution

RESOLUTION_INDICATORS = [
    "issue is resolved", "closing the ticket", "glad it's working",
    "great to hear", "happy to hear", "pleased to hear",
    "back up", "working now", "resolved", "fixed", "glad to hear",
    "have a good day", "resume normal", "anything else",
    "you're welcome", "welcome, have", "without any issues",
    "functioning as expected", "normal operations"
]

PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

EXIT_PHRASES = ["exit", "quit", "goodbye", "bye", "thank you bye"]

FAILURE_INDICATORS = [
    "not working", "still", "didn't work", "does not work",
    "same issue", "same problem", "still happening", "not fixed",
    "not resolved", "still down", "not helping", "still the same"
]

SOLUTION_INDICATORS = [
    "go to", "click", "restart", "open", "navigate", "check",
    "verify", "press", "unplug", "turn off", "log out", "clear"
]


def is_resolved_by_agent(response: str) -> bool:
    return any(phrase in response.lower() for phrase in RESOLUTION_INDICATORS)


def should_upgrade_priority(current: str, new: str) -> bool:
    return PRIORITY_ORDER.get(new, 4) < PRIORITY_ORDER.get(current, 4)


def is_failure_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in FAILURE_INDICATORS)


def is_solution_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in SOLUTION_INDICATORS)


def extract_name_via_llm(raw: str) -> str:
    if not raw or len(raw.strip()) < 2:
        return None

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only the person's name from the text. "
                    "Return just the name with correct capitalization, nothing else. "
                    "If no clear name is present, return UNKNOWN."
                )
            },
            {"role": "user", "content": raw}
        ],
        temperature=0.0,
        max_tokens=10
    )

    result = response.choices[0].message.content.strip()
    return None if result.upper() == "UNKNOWN" else result


def extract_store_id_via_llm(raw: str) -> str:
    if not raw or len(raw.strip()) < 1:
        return None

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only the store ID or store number from the text. "
                    "It could be a number like 341, or a code like MCD-0042. "
                    "If spoken as separate digits like 3 4 1, join them as 341. "
                    "Return just the store ID, nothing else. "
                    "If no clear store ID is present, return UNKNOWN."
                )
            },
            {"role": "user", "content": raw}
        ],
        temperature=0.0,
        max_tokens=15
    )

    result = response.choices[0].message.content.strip()
    return None if result.upper() == "UNKNOWN" else result


def detect_priority_from_rag(user_input: str) -> str:
    context = retrieve_solution(user_input)
    match = re.search(r'PRIORITY:\s*(P[1-4])', context)
    if match:
        return match.group(1)
    return "P3"


def get_caller_info() -> tuple:
    name_raw = listen_and_transcribe(duration=4)
    name = extract_name_via_llm(name_raw)

    if not name:
        speak("Sorry, could you repeat your name?")
        time.sleep(0.8)
        name_raw = listen_and_transcribe(duration=4)
        name = extract_name_via_llm(name_raw) or "Unknown"

    speak(f"Got it, {name}. Just to confirm, am I saying that right?")
    time.sleep(0.8)
    confirmation_raw = listen_and_transcribe(duration=4)

    if confirmation_raw and any(
        word in confirmation_raw.lower()
        for word in ["no", "wrong", "not", "incorrect", "nope"]
    ):
        speak("My apologies. Could you spell your name?")
        time.sleep(0.8)
        spelled_raw = listen_and_transcribe(duration=6)
        corrected = extract_name_via_llm(spelled_raw)
        if corrected:
            name = corrected

    speak(f"Thank you {name}. And your store ID?")
    time.sleep(0.8)
    store_raw = listen_and_transcribe(duration=4)
    store_id = extract_store_id_via_llm(store_raw)

    if not store_id:
        speak("Could you repeat just the store number?")
        time.sleep(0.8)
        store_raw = listen_and_transcribe(duration=4)
        store_id = extract_store_id_via_llm(store_raw) or "Unknown"

    speak(f"Store {store_id}, confirmed?")
    time.sleep(0.8)
    confirmation_raw = listen_and_transcribe(duration=4)

    if confirmation_raw and any(
        word in confirmation_raw.lower()
        for word in ["no", "wrong", "not", "incorrect", "nope"]
    ):
        speak("Please repeat your store number.")
        time.sleep(0.8)
        store_raw = listen_and_transcribe(duration=4)
        corrected = extract_store_id_via_llm(store_raw)
        if corrected:
            store_id = corrected

    return name, store_id


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
    failed_attempts = 0
    last_was_solution = False
    escalation_manager = None
    start_time = time.time()

    speak("Hello, thank you for calling McDonald's crew support. My name is Max. Who am I speaking with today?")
    time.sleep(0.8)

    caller_name, store_id = get_caller_info()

    conversation_history.append({
        "role": "system",
        "content": f"Caller name is {caller_name}. Store ID is {store_id}. Do not ask for these again."
    })

    speak(f"Perfect. Store {store_id}. Please describe your issue.")
    time.sleep(0.8)

    while True:
        user_input = listen_and_transcribe(duration=6)

        if not user_input or len(user_input.strip()) < 3:
            speak("I did not catch that. Please repeat.")
            continue

        if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
            if escalation_manager:
                from_agent = any(
                    is_resolved_by_agent(m["content"])
                    for m in conversation_history
                    if m["role"] == "assistant"
                )
                user_confirmed = any(
                    word in user_input.lower()
                    for word in ["resolved", "fixed", "working", "works", "done", "solved", "fine"]
                )
                resolved = "Resolved" if (from_agent or user_confirmed) else "Unresolved"
                finalize_and_log(
                    caller_name, store_id, conversation_history,
                    escalation_manager, resolved,
                    failed_attempts, start_time
                )
            speak("Thank you for calling. Have a good day.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        if escalation_manager is None:
            priority = detect_priority_from_rag(user_input)
            escalation_manager = EscalationManager(priority)
            speak(f"This is a {priority} priority issue. I will help you resolve it within {escalation_manager.sla_label}.")
        elif escalation_manager.priority != "P1" and len(conversation_history) <= 4:
            new_priority = detect_priority_from_rag(user_input)
            if should_upgrade_priority(escalation_manager.priority, new_priority):
                from escalation import SLA_LIMITS, SLA_LABELS
                escalation_manager.priority = new_priority
                escalation_manager.sla_limit = SLA_LIMITS[new_priority]
                escalation_manager.sla_label = SLA_LABELS[new_priority]
                speak(f"Upgrading this to {new_priority} priority based on what you described.")

        if last_was_solution and is_failure_response(user_input):
            failed_attempts += 1

        sla_warning = escalation_manager.get_sla_warning_message()
        if sla_warning:
            speak(sla_warning)

        should_escalate, reason = escalation_manager.should_escalate(
            failed_attempts, max_attempts=3
        )

        if should_escalate:
            escalation_manager.escalate(reason)
            speak(escalation_manager.get_escalation_message(reason))
            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Unresolved",
                failed_attempts, start_time
            )
            break

        response = get_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

        last_was_solution = is_solution_response(response)

        interrupted = speak(response)
        if interrupted:
            speak("Go ahead, I am listening.")

        if is_resolved_by_agent(response):
            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Resolved",
                failed_attempts, start_time
            )
            break


if __name__ == "__main__":
    run_voice_loop()