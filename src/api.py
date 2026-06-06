"""
api.py — McDonald's Crew Support Voice Agent
FastAPI backend that connects the frontend (index.html) to all existing
project modules: agent.py, rag.py, escalation.py, logger.py, stt.py

Run with:
    uvicorn api:app --reload --port 8000

Then open in Chrome:
    http://localhost:8000
    (do NOT open index.html directly as a file — Chrome blocks mic on file://)
"""

import re
import time
import uuid
import tempfile
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Import all your existing modules ──────────────────────────────────────────
from agent import get_response, extract_call_metadata, client, MODEL
from logger import log_call
from escalation import EscalationManager, SLA_LIMITS, SLA_LABELS
from rag import retrieve_solution

# ── Re-use the same helpers already written in voice_loop ─────────────────────
# We copy them here so api.py is standalone and voice_loop.py stays untouched.

RESOLUTION_INDICATORS = [
    "issue is resolved", "closing the ticket", "glad it's working",
    "great to hear that it", "happy to hear that it", "pleased to hear that it",
    "back up and running", "glad we could resolve", "happy we could help",
    "have a good day", "resume normal operations", "without any issues",
    "functioning as expected", "normal operations have resumed"
]

FAILURE_INDICATORS = [
    "not working", "still", "didn't work", "does not work",
    "same issue", "same problem", "still happening", "not fixed",
    "not resolved", "still down", "not helping", "still the same",
    # Real-world failure responses users actually say:
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

PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

EXIT_PHRASES = [
    "exit", "quit", "goodbye", "bye", "thank you bye",
    "don't want any help", "do not want any help",
    "no help", "not interested", "nothing", "that's all", "that's it",
    "thank you", "thanks"
]

# FIX #3: Explicit escalation request — user directly asks to escalate
ESCALATION_REQUEST_PHRASES = [
    "escalate", "escalation", "human agent", "real person", "actual person",
    "speak to someone", "talk to someone", "transfer me", "transfer this",
    "supervisor", "manager", "senior technician", "specialist",
    "please escalate", "want to escalate", "need to escalate",
    "raise a ticket", "create a ticket", "log a ticket", "open a ticket",
    "i want human", "connect me to"
]


# ── Helper functions ───────────────────────────────────────────────────────────

def is_resolved_by_agent(response: str) -> bool:
    return any(phrase in response.lower() for phrase in RESOLUTION_INDICATORS)

def is_failure_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in FAILURE_INDICATORS)

def is_solution_response(text: str) -> bool:
    return any(phrase in text.lower() for phrase in SOLUTION_INDICATORS)

def should_upgrade_priority(current: str, new: str) -> bool:
    return PRIORITY_ORDER.get(new, 4) < PRIORITY_ORDER.get(current, 4)

def detect_priority_from_rag(user_input: str) -> str:
    context = retrieve_solution(user_input)
    match = re.search(r'PRIORITY:\s*(P[1-4])', context)
    return match.group(1) if match else "P3"

def is_out_of_scope(text: str, conversation_history=None) -> bool:
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

def extract_name_via_llm(raw: str):
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

def extract_store_id_via_llm(raw: str):
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


# ── Session store (in-memory) ──────────────────────────────────────────────────
# Each session is a dict holding everything voice_loop.py would hold in variables.
# Key = session_id (UUID string), Value = session state dict.

sessions: dict = {}

def new_session_state() -> dict:
    """Creates a fresh session — mirrors run_voice_loop() initial state."""
    return {
        "conversation_history": [],
        "failed_attempts":      0,
        "out_of_scope_count":   0,
        "last_was_solution":    False,
        "escalation_manager":   None,
        "start_time":           time.time(),
        "caller_name":          None,
        "store_id":             None,
        "resolution_status":    "Unresolved",
        "ended":                False,
        # Tracks which onboarding step we're in before the main loop
        # Steps: "await_name" → "await_store" → "active"
        "onboarding_step":      "await_name",
    }


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="McDonald's Crew Support API")

# CORS — also needed for any external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX #4: Serve frontend/index.html via HTTP so Chrome grants persistent mic permission.
# Place index.html in ../frontend/ relative to this api.py file.
CURRENT_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = (CURRENT_DIR.parent / "frontend").resolve()

if not FRONTEND_DIR.exists():
    FRONTEND_DIR = (CURRENT_DIR / "frontend").resolve()

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_frontend():
    """Serve the frontend UI at the Root URL"""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": f"Frontend folder not found. Checked path: {FRONTEND_DIR}"}

# ── Request / Response models ──────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str
    message: str

class EndRequest(BaseModel):
    session_id: str


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Frontend pings this every 8 seconds to confirm backend is alive."""
    return {"status": "ok"}


@app.post("/session/start")
def session_start():
    """
    Called when user clicks Start Session in the frontend.
    Creates a new session and returns the greeting message.
    """
    session_id = str(uuid.uuid4())[:8].upper()
    sessions[session_id] = new_session_state()

    greeting = (
        "Hello, thank you for calling McDonald's crew support. "
        "My name is Max. Who am I speaking with today?"
    )
    return {
        "session_id": session_id,
        "response":   greeting
    }


@app.post("/session/message")
def session_message(req: MessageRequest):
    """
    Main endpoint. Called every time the user sends a text message.
    Mirrors the while-loop logic inside run_voice_loop().
    Returns Max's reply plus metadata the frontend needs to update the sidebar.
    """
    sid = req.session_id
    if sid not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    state = sessions[sid]

    if state["ended"]:
        raise HTTPException(status_code=400, detail="Session already ended.")

    user_input = req.message.strip()

    # ── ONBOARDING: collect name ───────────────────────────────────────────────
    if state["onboarding_step"] == "await_name":
        name = extract_name_via_llm(user_input) or user_input.strip().title() or "Crew Member"
        state["caller_name"] = name
        state["onboarding_step"] = "await_store"
        return _reply(state, f"Thank you {name}. And your store ID?")

    # ── ONBOARDING: collect store ID ───────────────────────────────────────────
    if state["onboarding_step"] == "await_store":
        store_id = extract_store_id_via_llm(user_input) or "Unknown"
        state["store_id"] = store_id

        # Inject caller context into history so LLM never asks again
        state["conversation_history"].append({
            "role": "system",
            "content": (
                f"Caller name is {state['caller_name']}. "
                f"Store ID is {store_id}. Do not ask for these again."
            )
        })
        state["onboarding_step"] = "active"
        return _reply(
            state,
            f"Store {store_id}, got it. Please describe your issue.",
            caller_name=state["caller_name"],
            store_id=store_id
        )

    # ── MAIN LOOP ──────────────────────────────────────────────────────────────

    caller_name = state["caller_name"] or "Crew Member"
    store_id    = state["store_id"]    or "Unknown"

    # Empty / too short
    if not user_input or len(user_input) < 3:
        return _reply(state, "I did not catch that. Could you please repeat?")

    # EXIT PHRASE CHECK
    if any(phrase in user_input.lower() for phrase in EXIT_PHRASES):
        em = state["escalation_manager"]
        res_status = "Unresolved"
        if em:
            from_agent = any(
                is_resolved_by_agent(m["content"])
                for m in state["conversation_history"]
                if m["role"] == "assistant"
            )
            user_confirmed = any(
                w in user_input.lower()
                for w in ["resolved", "fixed", "working", "works", "done", "solved", "fine", "thank"]
            )
            res_status = "Resolved" if (from_agent or user_confirmed) else "Unresolved"
            _finalize(state, res_status)
        state["ended"] = True
        # FIX #1: pass resolved flag so frontend updates call log correctly
        return _reply(
            state,
            "Thank you for calling. Have a good day!",
            ended=True,
            resolved=(res_status == "Resolved")
        )

    # OUT-OF-SCOPE CHECK
    if is_out_of_scope(user_input, state["conversation_history"]):
        state["out_of_scope_count"] += 1
        if state["out_of_scope_count"] >= 2:
            if state["escalation_manager"]:
                _finalize(state, "Unresolved")
            state["ended"] = True
            return _reply(
                state,
                "I can only assist with McDonald's IT and operations issues. "
                "I will close this call now. Please call back if you have a system "
                "related issue. Have a good day.",
                ended=True
            )
        return _reply(
            state,
            "I can only help with McDonald's store IT and operations issues. "
            "Do you have a system related problem I can assist with?"
        )

    # Valid in-scope message — reset counter
    state["out_of_scope_count"] = 0

    # Append user message
    state["conversation_history"].append({"role": "user", "content": user_input})

    # PRIORITY DETECTION
    extra_speak = None
    if state["escalation_manager"] is None:
        priority = detect_priority_from_rag(user_input)
        state["escalation_manager"] = EscalationManager(priority)
        extra_speak = f"No worries, {caller_name}. I will help you resolve this as soon as possible."
    elif (
        state["escalation_manager"].priority != "P1"
        and len(state["conversation_history"]) <= 4
    ):
        new_priority = detect_priority_from_rag(user_input)
        if should_upgrade_priority(state["escalation_manager"].priority, new_priority):
            state["escalation_manager"].priority   = new_priority
            state["escalation_manager"].sla_limit  = SLA_LIMITS[new_priority]
            state["escalation_manager"].sla_label  = SLA_LABELS[new_priority]
            extra_speak = f"Upgrading this to {new_priority} priority based on what you described."

    em = state["escalation_manager"]

    # FIX #3: Explicit escalation request — user directly asked to escalate
    if any(phrase in user_input.lower() for phrase in ESCALATION_REQUEST_PHRASES):
        em.escalate("user_requested")
        esc_msg = (
            f"Understood, {caller_name}. I am escalating this to a senior technician right away. "
            f"You will be contacted shortly. Your ticket has been logged. Have a good day."
        )
        _finalize(state, "Unresolved")
        state["ended"] = True
        return _reply(
            state, esc_msg,
            priority=em.priority,
            failed_attempts=state["failed_attempts"],
            escalated=True,
            ended=True
        )

    # FIX #2: Increment failed_attempts — now catches "already done/tried/checked" etc.
    if state["last_was_solution"] and is_failure_response(user_input):
        state["failed_attempts"] += 1

    # SLA WARNING
    sla_warning = em.get_sla_warning_message()

    # ESCALATION CHECK — auto-escalate after 3 failed attempts
    should_esc, reason = em.should_escalate(state["failed_attempts"], max_attempts=3)
    if should_esc:
        em.escalate(reason)
        esc_msg = em.get_escalation_message(reason)
        _finalize(state, "Unresolved")
        state["ended"] = True
        return _reply(
            state, esc_msg,
            priority=em.priority,
            failed_attempts=state["failed_attempts"],
            escalated=True,
            ended=True
        )

    # GET AGENT RESPONSE
    agent_response = get_response(state["conversation_history"])
    state["conversation_history"].append({"role": "assistant", "content": agent_response})
    state["last_was_solution"] = is_solution_response(agent_response)

    # Build final reply — prepend extra_speak and sla_warning if present
    full_response = ""
    if extra_speak:
        full_response += extra_speak + " "
    if sla_warning:
        full_response += sla_warning + " "
    full_response += agent_response

    # RESOLUTION CHECK
    resolved = False
    if is_resolved_by_agent(agent_response):
        closing = (
            f"Glad I could help, {caller_name}. "
            "Thank you for calling McDonald's crew support. Have a great day!"
        )
        full_response = full_response.strip() + " " + closing
        _finalize(state, "Resolved")
        state["ended"] = True
        resolved = True

    return _reply(
        state,
        full_response.strip(),
        priority=em.priority,
        failed_attempts=state["failed_attempts"],
        caller_name=caller_name,
        store_id=store_id,
        resolved=resolved,
        ended=resolved
    )


@app.post("/session/end")
def session_end(req: EndRequest):
    """
    Called when user clicks End Call in the frontend.
    Logs the call and cleans up the session.
    """
    sid = req.session_id
    if sid not in sessions:
        return {"status": "ok", "message": "Session not found (already cleaned up)."}

    state = sessions[sid]
    if not state["ended"] and state["escalation_manager"]:
        _finalize(state, state["resolution_status"])

    sessions.pop(sid, None)
    return {"status": "ok", "message": "Session ended and logged."}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Receives audio blob from the mic button in the frontend.
    Sends it to Groq Whisper and returns the transcribed text.
    """
    try:
        # Save uploaded audio to a temp file
        suffix = os.path.splitext(audio.filename or "recording.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await audio.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Send to Groq Whisper (same model as stt.py)
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text"
            )

        os.unlink(tmp_path)

        # Groq may return a string or an object
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return {"text": text.strip()}

    except Exception as e:
        return {"text": "", "error": str(e)}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _finalize(state: dict, resolution_status: str):
    """Logs the call to calls.csv. Mirrors finalize_and_log() in voice_loop.py."""
    em = state["escalation_manager"]
    if not em:
        return
    try:
        metadata = extract_call_metadata(state["conversation_history"])
        duration = time.time() - state["start_time"]
        log_call(
            caller_name       = state["caller_name"]  or "Unknown",
            store_id          = state["store_id"]     or "Unknown",
            issue_description = metadata.get("issue_description", "N/A"),
            priority          = em.priority,
            resolution_status = resolution_status,
            escalated_to_human= em.escalated,
            attempts          = state["failed_attempts"],
            duration_seconds  = duration
        )
        state["resolution_status"] = resolution_status
        print(f"[api] Call logged — {resolution_status} | {em.priority} | "
              f"{state['caller_name']} | Store {state['store_id']}")
    except Exception as e:
        print(f"[api] Logging error: {e}")


def _reply(state: dict, message: str, **kwargs) -> dict:
    """Builds the standard JSON response the frontend expects."""
    em = state["escalation_manager"]
    return {
        "response":       message,
        "priority":       kwargs.get("priority",       em.priority if em else None),
        "failed_attempts":kwargs.get("failed_attempts",state["failed_attempts"]),
        "caller_name":    kwargs.get("caller_name",    state["caller_name"]),
        "store_id":       kwargs.get("store_id",       state["store_id"]),
        "escalated":      kwargs.get("escalated",      em.escalated if em else False),
        "resolved":       kwargs.get("resolved",       False),
        "ended":          kwargs.get("ended",          False),
    }