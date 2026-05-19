import csv
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "calls.csv")

HEADERS = [
    "call_id",
    "timestamp",
    "caller_name",
    "store_id",
    "issue_description",
    "priority",
    "resolution_status",
    "escalated_to_human",
    "attempts",
    "duration_seconds"
]


def init_log():
    """
    Creates the CSV file with headers if it does not exist.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
        print("Log file created.")


def generate_call_id() -> str:
    """
    Generates a unique call ID based on timestamp.
    """
    return f"CALL-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def log_call(
    caller_name: str,
    store_id: str,
    issue_description: str,
    priority: str,
    resolution_status: str,
    escalated_to_human: bool,
    attempts: int,
    duration_seconds: float
):
    """
    Appends a single call record to the CSV log.
    """
    init_log()

    record = {
        "call_id": generate_call_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "caller_name": caller_name,
        "store_id": store_id,
        "issue_description": issue_description,
        "priority": priority,
        "resolution_status": resolution_status,
        "escalated_to_human": escalated_to_human,
        "attempts": attempts,
        "duration_seconds": round(duration_seconds, 2)
    }

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(record)

    print(f"Call logged: {record['call_id']} | Priority: {priority} | Resolved: {resolution_status} | Escalated: {escalated_to_human}")


if __name__ == "__main__":
    log_call(
        caller_name="John",
        store_id="MCD-0042",
        issue_description="Inventory not updating on back office system",
        priority="P2",
        resolution_status="Resolved",
        escalated_to_human=False,
        attempts=2,
        duration_seconds=134.5
    )

    log_call(
        caller_name="Sarah",
        store_id="MCD-0087",
        issue_description="Store system completely down",
        priority="P1",
        resolution_status="Unresolved",
        escalated_to_human=True,
        attempts=3,
        duration_seconds=210.0
    )