import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import get_response, extract_call_metadata
from escalation import EscalationManager
from logger import log_call, LOG_PATH


def test_agent_responds():
    history = [{"role": "user", "content": "Our inventory is not updating on the back office system."}]
    response = get_response(history)
    assert isinstance(response, str)
    assert len(response) > 10
    print(f"PASS test_agent_responds: {response[:80]}...")


def test_agent_maintains_context():
    history = [
        {"role": "user", "content": "Our screen is not showing menu prices."},
        {"role": "assistant", "content": "Please go to the POS admin panel and click Sync Now."},
        {"role": "user", "content": "We tried that but it still is not working."}
    ]
    response = get_response(history)
    assert isinstance(response, str)
    assert len(response) > 10
    print(f"PASS test_agent_maintains_context: {response[:80]}...")


def test_metadata_extraction():
    history = [
        {"role": "user", "content": "Our store system is completely down. No transactions are going through."},
        {"role": "assistant", "content": "This is a P1 issue. Please check the main server power and restart it."},
        {"role": "user", "content": "We restarted but it is still down."}
    ]
    metadata = extract_call_metadata(history)
    assert "priority" in metadata
    assert "resolution_status" in metadata
    assert "issue_description" in metadata
    assert metadata["priority"] in ["P1", "P2", "P3", "P4"]
    print(f"PASS test_metadata_extraction: {metadata}")


def test_escalation_by_attempts():
    manager = EscalationManager("P1")
    should, reason = manager.should_escalate(attempts=3, max_attempts=3)
    assert should is True
    assert reason == "max_attempts_reached"
    print(f"PASS test_escalation_by_attempts: escalated={should}, reason={reason}")


def test_escalation_p1_sla_label():
    manager = EscalationManager("P1")
    assert manager.sla_label == "30 minutes"
    assert manager.sla_limit == 1800
    print(f"PASS test_escalation_p1_sla_label: sla_limit={manager.sla_limit}s")


def test_escalation_not_triggered_early():
    manager = EscalationManager("P2")
    should, reason = manager.should_escalate(attempts=1, max_attempts=3)
    assert should is False
    assert reason is None
    print(f"PASS test_escalation_not_triggered_early: escalate={should}")


def test_csv_logger():
    log_call(
        caller_name="Test User",
        store_id="MCD-TEST",
        issue_description="Test issue for unit test",
        priority="P3",
        resolution_status="Resolved",
        escalated_to_human=False,
        attempts=1,
        duration_seconds=45.0
    )
    assert os.path.exists(LOG_PATH)
    with open(LOG_PATH, "r") as f:
        content = f.read()
    assert "Test User" in content
    assert "MCD-TEST" in content
    print(f"PASS test_csv_logger: log entry written successfully.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    tests = [
        test_escalation_by_attempts,
        test_escalation_p1_sla_label,
        test_escalation_not_triggered_early,
        test_csv_logger,
        test_metadata_extraction,
        test_agent_responds,
        test_agent_maintains_context,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1

    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")