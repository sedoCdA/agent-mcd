import re
import time

from stt import listen_and_transcribe
from tts import speak
from agent import get_response, extract_call_metadata, client, MODEL
from logger import log_call
from escalation import EscalationManager
from rag import retrieve_solution

# ── Constants — kept in sync with api.py ──────────────────────────────────────

RESOLUTION_INDICATORS = [
    "issue is resolved", "closing the ticket", "glad it's working",
    "great to hear that it", "happy to hear that it", "pleased to hear that it",
    "back up and running", "glad we could resolve", "happy we could help",
    "have a good day", "resume normal operations", "without any issues",
    "functioning as expected", "normal operations have resumed"
]

PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

EXIT_PHRASES = [
    "exit", "quit", "goodbye", "bye", "thank you bye",
    "don't want any help", "do not want any help",
    "no help", "not interested", "nothing", "that's all", "that's it",
    "thank you", "thanks"
]

# SYNC: Expanded to match api.py — covers what users actually say when a solution fails
FAILURE_INDICATORS = [
    "not working", "still", "didn't work", "does not work",
    "same issue", "same problem", "still happening", "not fixed",
    "not resolved", "still down", "not helping", "still the same",
    "already done", "already tried", "already checked", "already restarted",
    "already rebooted", "already cleared", "already verified", "already tested",
    "no luck", "no result", "no change", "didn't help", "did not help",
    "doesn't help", "does not help", "not helpful", "tried that",
    "tried this", "tried everything", "nothing works", "nothing worked",
    "still same", "same thing",
]

SOLUTION_INDICATORS = [
    "go to", "click", "restart", "open", "navigate", "check",
    "verify", "press", "unplug", "turn off", "log out", "clear"
]

# SYNC: Matches api.py — user explicitly asks to be escalated to a human
ESCALATION_REQUEST_PHRASES = [
    "escalate", "escalation", "human agent", "real person", "actual person",
    "speak to someone", "talk to someone", "transfer me", "transfer this",
    "supervisor", "manager", "senior technician", "specialist",
    "please escalate", "want to escalate", "need to escalate",
    "raise a ticket", "create a ticket", "log a ticket", "open a ticket",
    "i want human", "connect me to"
]

OUT_OF_SCOPE_LIMIT = 2


# ── Helper functions ───────────────────────────────────────────────────────────

def is_resolved_by_agent(response: str) -> bool:
    return any(phrase in response.lower() for phrase in RESOLUTION_INDICATORS)


def should_upgrade_priority(current: str, new: str) -> bool:
    return PRIORITY_ORDER.get(new, 4) < PRIORITY_ORDER.get(current, 4)


def is_failure_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in FAILURE_INDICATORS)


def is_solution_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in SOLUTION_INDICATORS)


def is_escalation_request(text: str) -> bool:
    """Returns True if the caller explicitly asks to be escalated to a human."""
    return any(phrase in text.lower() for phrase in ESCALATION_REQUEST_PHRASES)


def is_valid_name(name: str) -> bool:
    """
    Sanity-checks extracted names.
    A real person's name is 1–3 words, each at least 2 alphabetic characters.
    Rejects sentence fragments that Whisper picks up from TTS echo.
    """
    if not name:
        return False
    words = name.strip().split()
    if not (1 <= len(words) <= 3):
        return False
    for w in words:
        clean = w.replace("'", "").replace("-", "")
        if len(clean) < 2 or not clean.isalpha():
            return False
    return True


def is_out_of_scope(text: str, conversation_history=None) -> bool:
    """
    Detects if the user message has nothing to do with IT or store operations.
    Uses LLM for accuracy but avoids false positives during follow-up troubleshooting.
    """
    lowered = text.lower().strip()

    followup_indicators = [
        "yes", "yeah", "yep", "no", "nope", "same", "still same",
        "already tried", "still", "not working", "same issue",
        "same problem", "i checked", "we checked", "they checked", "cache",
        "connection is stable", "password is correct", "tried that as well",
    ]
    if any(word in lowered for word in followup_indicators):
        return False

    if conversation_history:
        recent_context = " ".join(
            m["content"].lower()
            for m in conversation_history[-6:]
            if m["role"] in ["user", "assistant"]
        )
        context_keywords = [
            "login", "website", "browser", "cache", "cookie", "password",
            "account", "portal", "network", "internet", "wifi", "system",
            "crew", "store", "app", "server", "pos", "printer", "router",
            "screen", "display", "restart", "reboot", "boot", "power",
            "cable", "connection", "safe mode", "diagnostic"
        ]
        if any(keyword in recent_context for keyword in context_keywords):
            return False

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classifier. Determine if the user message is related to "
                    "McDonald's store IT or operations support. "
                    "Reply with only YES if it is related, or NO if it is not related. "
                    "Nothing else."
                )
            },
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        max_tokens=5
    )
    result = response.choices[0].message.content.strip().upper()
    return result == "NO"


def extract_name_via_llm(raw: str) -> str:
    if not raw or len(raw.strip()) < 2:
        return None

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only the person's first and last name from the text. "
                    "Return just the name with correct capitalization, nothing else. "
                    "A name is 1 to 3 words maximum. "
                    "If no clear person's name is present, return UNKNOWN."
                )
            },
            {"role": "user", "content": raw}
        ],
        temperature=0.0,
        max_tokens=10
    )

    result = response.choices[0].message.content.strip()
    if result.upper() == "UNKNOWN":
        return None

    # Sanity check: reject anything that doesn't look like a real name
    # (catches TTS echo fragments like "So We Will Try To Give")
    if not is_valid_name(result):
        return None

    return result


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
    """
    Collects caller name and store ID via STT + LLM extraction.
    Always waits 1.2s before each listen to prevent TTS echo bleed into recording.
    """
    # ── NAME ──────────────────────────────────────────────────────────────────
    # Wait 1.2s before first recording — project spec requires this before every listen,
    # and it prevents Whisper from picking up the tail of Max's greeting via speaker echo.
    time.sleep(1.2)
    name_raw = listen_and_transcribe(duration=4)
    name = extract_name_via_llm(name_raw)

    if not name:
        speak("Sorry, I didn't catch your name. Could you repeat it?")
        time.sleep(1.2)
        name_raw = listen_and_transcribe(duration=4)
        name = extract_name_via_llm(name_raw) or "Unknown"

    speak(f"Got it, {name}. Just to confirm, am I saying that right?")
    time.sleep(1.2)
    confirmation_raw = listen_and_transcribe(duration=4)

    if confirmation_raw and any(
        word in confirmation_raw.lower()
        for word in ["no", "wrong", "not", "incorrect", "nope"]
    ):
        speak("My apologies. Could you spell your name for me?")
        time.sleep(1.2)
        spelled_raw = listen_and_transcribe(duration=6)
        corrected = extract_name_via_llm(spelled_raw)
        if corrected:
            name = corrected

    # ── STORE ID ──────────────────────────────────────────────────────────────
    speak(f"Thank you {name}. And your store ID?")
    time.sleep(1.2)
    store_raw = listen_and_transcribe(duration=4)
    store_id = extract_store_id_via_llm(store_raw)

    if not store_id:
        speak("Could you repeat just the store number?")
        time.sleep(1.2)
        store_raw = listen_and_transcribe(duration=4)
        store_id = extract_store_id_via_llm(store_raw) or "Unknown"

    speak(f"Store {store_id}, confirmed?")
    time.sleep(1.2)
    confirmation_raw = listen_and_transcribe(duration=4)

    if confirmation_raw and any(
        word in confirmation_raw.lower()
        for word in ["no", "wrong", "not", "incorrect", "nope"]
    ):
        # Try to extract the store ID from the correction itself first
        # e.g. "No, it's 721" — no need to re-listen
        corrected = extract_store_id_via_llm(confirmation_raw)
        if not corrected:
            speak("Please repeat your store number.")
            time.sleep(1.2)
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
    failed_attempts      = 0
    out_of_scope_count   = 0
    last_was_solution    = False
    escalation_manager   = None
    start_time           = time.time()

    speak("Hello, thank you for calling McDonald's crew support. My name is Max. Who am I speaking with today?")
    time.sleep(2.5)

    caller_name, store_id = get_caller_info()

    conversation_history.append({
        "role": "system",
        "content": f"Caller name is {caller_name}. Store ID is {store_id}. Do not ask for these again."
    })

    speak(f"Perfect. Store {store_id}. Please describe your issue.")
    time.sleep(2.0)

    while True:
        time.sleep(1.2)   # project spec: 1.2s before every listen
        user_input = listen_and_transcribe(duration=6)

        if not user_input or len(user_input.strip()) < 3:
            speak("I did not catch that. Please repeat.")
            continue

        # ── EXIT PHRASE CHECK ──────────────────────────────────────────────────
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
                    escalation_manager, resolved, failed_attempts, start_time
                )
            speak("Thank you for calling. Have a good day.")
            break

        # ── OUT-OF-SCOPE CHECK ─────────────────────────────────────────────────
        if is_out_of_scope(user_input, conversation_history):
            out_of_scope_count += 1
            if out_of_scope_count >= OUT_OF_SCOPE_LIMIT:
                speak(
                    "I can only assist with McDonald's IT and operations issues. "
                    "I will close this call now. Please call back if you have a system "
                    "related issue. Have a good day."
                )
                if escalation_manager:
                    finalize_and_log(
                        caller_name, store_id, conversation_history,
                        escalation_manager, "Unresolved", failed_attempts, start_time
                    )
                break
            speak(
                "I can only help with McDonald's store IT and operations issues. "
                "Do you have a system related problem I can assist with?"
            )
            continue

        # Valid in-scope message — reset counter
        out_of_scope_count = 0
        conversation_history.append({"role": "user", "content": user_input})

        # ── PRIORITY DETECTION ─────────────────────────────────────────────────
        if escalation_manager is None:
            priority = detect_priority_from_rag(user_input)
            escalation_manager = EscalationManager(priority)
            speak(f"No worries, {caller_name}. I will help you resolve this as soon as possible.")
        elif escalation_manager.priority != "P1" and len(conversation_history) <= 4:
            new_priority = detect_priority_from_rag(user_input)
            if should_upgrade_priority(escalation_manager.priority, new_priority):
                from escalation import SLA_LIMITS, SLA_LABELS
                escalation_manager.priority   = new_priority
                escalation_manager.sla_limit  = SLA_LIMITS[new_priority]
                escalation_manager.sla_label  = SLA_LABELS[new_priority]
                speak(f"Upgrading this to {new_priority} priority based on what you described.")

        # ── EXPLICIT ESCALATION REQUEST ────────────────────────────────────────
        # SYNC with api.py: caller says "escalate", "human", "supervisor" etc.
        if is_escalation_request(user_input):
            escalation_manager.escalate("user_requested")
            speak(
                f"Understood, {caller_name}. I am escalating this to a senior technician right away. "
                f"You will be contacted shortly. Your ticket has been logged. Have a good day."
            )
            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Unresolved", failed_attempts, start_time
            )
            break

        # ── FAILED ATTEMPTS — expanded indicators match api.py ─────────────────
        if last_was_solution and is_failure_response(user_input):
            failed_attempts += 1

        # ── SLA WARNING ────────────────────────────────────────────────────────
        sla_warning = escalation_manager.get_sla_warning_message()
        if sla_warning:
            speak(sla_warning)

        # ── AUTO-ESCALATION after 3 failed attempts ────────────────────────────
        should_esc, reason = escalation_manager.should_escalate(
            failed_attempts, max_attempts=3
        )
        if should_esc:
            escalation_manager.escalate(reason)
            speak(escalation_manager.get_escalation_message(reason))
            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Unresolved", failed_attempts, start_time
            )
            break

        # ── AGENT RESPONSE ─────────────────────────────────────────────────────
        response = get_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        last_was_solution = is_solution_response(response)

        interrupted = speak(response)
        if interrupted:
            speak("Go ahead, I am listening.")

        # ── RESOLUTION CHECK ───────────────────────────────────────────────
        if is_resolved_by_agent(response):
            closing = (
                f"Glad I could help, {caller_name}. "
                "Thank you for calling McDonald's crew support. Have a great day!"
            )
            speak(closing)
            finalize_and_log(
                caller_name, store_id, conversation_history,
                escalation_manager, "Resolved", failed_attempts, start_time
            )
            break


if __name__ == "__main__":
    run_voice_loop()