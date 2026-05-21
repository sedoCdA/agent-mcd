import time
from datetime import datetime


SLA_LIMITS = {
    "P1": 30 * 60,
    "P2": 60 * 60,
    "P3": 4 * 60 * 60,
    "P4": 24 * 60 * 60
}

SLA_LABELS = {
    "P1": "30 minutes",
    "P2": "60 minutes",
    "P3": "4 hours",
    "P4": "24 hours"
}


class EscalationManager:
    def __init__(self, priority: str):
        self.priority = priority.upper()
        self.start_time = time.time()
        self.escalated = False
        self.escalation_reason = None
        self.sla_limit = SLA_LIMITS.get(self.priority, SLA_LIMITS["P3"])
        self.sla_label = SLA_LABELS.get(self.priority, "4 hours")

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def is_sla_breached(self) -> bool:
        return self.elapsed_seconds() >= self.sla_limit

    def remaining_seconds(self) -> float:
        remaining = self.sla_limit - self.elapsed_seconds()
        return max(0.0, remaining)

    def should_escalate(self, attempts: int, max_attempts: int = 3) -> tuple:
        """
        Returns (should_escalate: bool, reason: str)
        Escalates if max attempts reached or SLA is breached.
        """
        if attempts >= max_attempts:
            return True, "max_attempts_reached"

        if self.is_sla_breached():
            return True, "sla_breached"

        return False, None

    def escalate(self, reason: str):
        self.escalated = True
        self.escalation_reason = reason

    def get_sla_warning_message(self) -> str:
        """
        Returns a spoken warning when SLA is close to breaching.
        Warns when 20 percent of SLA time remains.
        """
        remaining = self.remaining_seconds()
        threshold = self.sla_limit * 0.20

        if remaining <= threshold and not self.is_sla_breached():
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            if minutes > 0:
                return f"Please note this is a {self.priority} issue and we have {minutes} minutes remaining to resolve it."
            else:
                return f"Please note we have only {seconds} seconds remaining within the SLA for this {self.priority} issue."

        return None

    def get_escalation_message(self, reason: str) -> str:
        if reason == "sla_breached":
            return (
                f"I am sorry, but this {self.priority} issue has exceeded its SLA of {self.sla_label}. "
                f"I am escalating this to a human agent immediately. Someone will contact you shortly."
            )
        elif reason == "max_attempts_reached":
            return (
                f"I have attempted to resolve this issue as per my knowledge. "
                f"I am now escalating this to a human agent. Please stay available."
            )
        return "I am escalating this issue to a human agent who will assist you shortly."

    def summary(self) -> dict:
        return {
            "priority": self.priority,
            "sla_limit_seconds": self.sla_limit,
            "elapsed_seconds": round(self.elapsed_seconds(), 2),
            "sla_breached": self.is_sla_breached(),
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason
        }


if __name__ == "__main__":
    print("Testing P1 escalation by attempts:")
    manager = EscalationManager("P1")
    time.sleep(1)

    for attempt in range(1, 5):
        should, reason = manager.should_escalate(attempt)
        print(f"Attempt {attempt}: escalate={should}, reason={reason}")
        if should:
            manager.escalate(reason)
            print(manager.get_escalation_message(reason))
            break

    print()
    print("Summary:", manager.summary())