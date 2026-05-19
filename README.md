# McDonald's Crew Support Voice Agent

An agentic AI voice assistant that handles internal IT and operations 
support calls from McDonald's crew members. Replaces redundant human 
support calls with a real-time voice bot.

## Features

- Real-time voice input and output with interrupt detection
- RAG over internal solution documents using FAISS
- Priority classification P1 through P4 with SLA tracking
- Auto escalation when max attempts or SLA is breached
- Full CSV logging of every call with metadata extraction

## Tech Stack

- STT: Groq Whisper large-v3
- LLM: Groq llama-3.3-70b-versatile
- RAG: FAISS + LangChain + sentence-transformers
- TTS: Microsoft Edge Neural TTS
- Logging: Python CSV

## Setup

    git clone https://github.com/YOUR_USERNAME/mcds-voice-agent
    cd mcds-voice-agent
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env
    # Add your GROQ_API_KEY to .env

## Run

    python src/main.py

## Test

    python tests/test_agent.py

## Project Structure

    src/
      main.py         Entry point with startup checks
      voice_loop.py   Core conversation loop
      agent.py        Groq LLM with RAG injection
      rag.py          FAISS vector store and retrieval
      stt.py          Groq Whisper speech to text
      tts.py          Edge TTS with interrupt detection
      escalation.py   SLA tracking and escalation logic
      logger.py       CSV call logging
    data/
      solutions.txt   Knowledge base documents
    logs/
      calls.csv       Call history and metadata
    tests/
      test_agent.py   Unit tests