import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from voice_loop import run_voice_loop
from rag import build_vector_store, VECTOR_STORE_PATH
from logger import init_log


def check_environment():
    """
    Verifies all required environment variables and files are present
    before starting the agent.
    """
    errors = []

    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY is missing from your .env file.")

    solutions_path = os.path.join(os.path.dirname(__file__), "..", "data", "solutions.txt")
    if not os.path.exists(solutions_path):
        errors.append("data/solutions.txt not found. Add your solution documents.")

    if errors:
        print("Startup failed. Fix the following issues:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def setup():
    """
    Initializes all required components before the agent starts.
    Builds vector store if not already built.
    Initializes log file.
    """
    print("Initializing McDonald's Crew Support Agent...")

    check_environment()

    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Building from solution documents...")
        build_vector_store()
    else:
        print("Vector store found. Skipping rebuild.")

    init_log()
    print("All systems ready. Starting agent.\n")
    print("-" * 50)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup()
    run_voice_loop()