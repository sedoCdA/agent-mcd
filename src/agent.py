import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are Max, a voice support agent for McDonald's crew IT issues.

Rules:
- Reply in 1 to 2 short sentences only. Never more.
- No bullet points, no lists, no markdown, no special characters.
- Be direct. Skip pleasantries after the first message.
- Ask only one thing at a time.
- Never ask for the caller name or store ID. That is already known.
- Never decide to escalate on your own. Never say you are escalating.
- Never mention attempt limits or escalation in your responses.
- The system handles escalation automatically. Your job is only to solve the issue.
- Provide one concrete solution step per response based on the knowledge base.
- If the step did not work, try the next step from the knowledge base.

Priority levels for your awareness only, do not mention them unless asked:
P1: store down, 30 minutes
P2: major system issue, 60 minutes
P3: partial issue, 4 hours
P4: minor issue, 24 hours

- When the issue is confirmed resolved by the caller, always end with:
  "Glad I could help. Thank you for calling McDonald's crew support. Have a great day!"
  This signals the system to close the call cleanly.
  
Tone: calm, fast, professional.
"""


def get_response(conversation_history: list) -> str:
    """
    Sends the full conversation history to Groq LLM and returns the assistant reply.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def extract_call_metadata(conversation_history: list) -> dict:
    """
    Asks the LLM to extract structured metadata from the conversation.
    Returns priority, resolution status, and a short issue description.
    """
    transcript = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history]
    )

    prompt = f"""
Based on this support call transcript, extract the following in JSON format only, no explanation:
{{
  "priority": "P1 or P2 or P3 or P4",
  "resolution_status": "Resolved or Unresolved",
  "issue_description": "one sentence summary of the issue"
}}

Transcript:
{transcript}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    import json
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {
            "priority": "P3",
            "resolution_status": "Unresolved",
            "issue_description": "Issue details unavailable"
        }


if __name__ == "__main__":
    history = []

    test_inputs = [
        "Hi, our screen is not showing the updated menu prices.",
        "We already tried restarting it but it did not work.",
        "The issue is still there, nothing is working."
    ]

    for user_text in test_inputs:
        print(f"Crew: {user_text}")
        history.append({"role": "user", "content": user_text})
        reply = get_response(history)
        print(f"Mac: {reply}")
        history.append({"role": "assistant", "content": reply})
        print()