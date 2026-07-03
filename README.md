---
title: Agent MCD
emoji: 🍔
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🍔 Agent MCD - McDonald's Crew Support Voice Agent

> An agentic AI voice assistant that handles real-time IT and operations support calls from McDonald's crew members. Replaces repetitive human support calls with a voice bot that listens, diagnoses, escalates, and logs — end to end.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)
![Whisper](https://img.shields.io/badge/STT-Groq%20Whisper%20large--v3-purple)
![FAISS](https://img.shields.io/badge/RAG-FAISS%20%2B%20LangChain-yellow)
![Docker](https://img.shields.io/badge/Deploy-Docker%20%2F%20HuggingFace%20Spaces-blue?logo=docker)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Agent](#running-the-agent)
- [API Reference](#api-reference)
- [RAG Knowledge Base](#rag-knowledge-base)
- [Priority & Escalation System](#priority--escalation-system)
- [Call Logging](#call-logging)
- [Frontend](#frontend)
- [Docker & Deployment](#docker--deployment)
- [Running Tests](#running-tests)
- [How a Call Works](#how-a-call-works)
- [Environment Variables](#environment-variables)

---

## Overview

Agent MCD is a full-stack AI voice support agent built for McDonald's crew members to report and resolve store IT issues without waiting for a human support representative. The agent, named **Max**, conducts a structured support call: it collects the caller's name and store ID, understands the issue, retrieves relevant step-by-step solutions from a FAISS vector store, tracks resolution attempts, and automatically escalates to a human technician when solutions are exhausted or an SLA is about to breach.

The system ships with two interaction modes:

- **Voice loop** (`src/main.py`) — runs on a local machine with a microphone and speaker for fully hands-free operation.
- **Web API + browser frontend** (`src/api.py` + `frontend/index.html`) — exposes a FastAPI backend and a browser-based UI, suitable for deployment on Hugging Face Spaces or any cloud host.

---

## Features

| Feature | Details |
|---|---|
| 🎙️ Real-time voice I/O | Microphone recording + Edge Neural TTS with interrupt detection |
| 🧠 LLM-powered conversation | Groq LLaMA 3.3 70B with strict 1–2 sentence response rules |
| 🔍 RAG over internal docs | FAISS + LangChain + `sentence-transformers/all-MiniLM-L6-v2` |
| 🚨 Priority classification | P1–P4 auto-detected from issue description via RAG context |
| ⏱️ SLA tracking & warnings | Per-priority SLA timers with spoken warnings at 20% remaining |
| 🔼 Auto-escalation | Triggers after 3 failed attempts or SLA breach; also on explicit user request |
| 📋 CSV call logging | Every call logged with ID, timestamp, priority, resolution status, duration |
| 🌐 Browser UI | Single-page HTML frontend with mic recording via Web API |
| 🚀 CI/CD to HuggingFace | GitHub Actions pipeline syncs main branch to HF Spaces on every push |
| 🐳 Docker-ready | Dockerfile included, runs on port 7860 (HF-compatible) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User (Voice / Browser)                      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ Audio / Text
                       ▼
          ┌────────────────────────┐
          │   STT (Groq Whisper)   │  ◄── stt.py / /transcribe endpoint
          └────────────┬───────────┘
                       │ Transcribed text
                       ▼
          ┌────────────────────────┐     ┌──────────────────────────┐
          │   Voice Loop / API     │────►│  Out-of-scope Classifier  │
          │  (voice_loop.py /      │     │  (Groq LLM YES/NO)        │
          │   api.py)              │     └──────────────────────────┘
          └────────┬───────────────┘
                   │
        ┌──────────┴───────────────┐
        │                          │
        ▼                          ▼
┌───────────────┐        ┌─────────────────────┐
│  RAG Engine   │        │  Escalation Manager  │
│  (rag.py)     │        │  (escalation.py)     │
│  FAISS index  │        │  SLA + attempts      │
└───────┬───────┘        └──────────┬──────────┘
        │ Context                    │ Escalate?
        ▼                           ▼
┌────────────────────────────────────────────────┐
│             Groq LLaMA 3.3 70B                 │
│          System prompt "Max" persona           │
│          (agent.py → get_response)             │
└──────────────────────┬─────────────────────────┘
                       │ Agent reply
                       ▼
          ┌────────────────────────┐
          │   TTS (Edge Neural)    │  ◄── tts.py / browser TTS
          └────────────┬───────────┘
                       │ Audio output
                       ▼
          ┌────────────────────────┐
          │   Logger (logger.py)   │  → logs/calls.csv
          └────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language Model | Groq `llama-3.3-70b-versatile` |
| Speech-to-Text | Groq `whisper-large-v3` |
| Text-to-Speech | Microsoft Edge Neural TTS (`edge-tts`, voice `en-US-AriaNeural`) |
| Vector Store | FAISS (CPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| RAG Framework | LangChain Community + LangChain HuggingFace |
| Web Framework | FastAPI + Uvicorn |
| Audio I/O | `sounddevice`, `soundfile`, `numpy` |
| Containerization | Docker (`python:3.10-slim`) |
| CI/CD | GitHub Actions → Hugging Face Spaces |

---

## Project Structure

```
agent-mcd/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD: GitHub → Hugging Face Spaces
│
├── data/
│   ├── faiss_index/            # Pre-built FAISS vector store
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── solutions.txt           # Knowledge base (ISSUE / PRIORITY / SOLUTION blocks)
│
├── frontend/
│   └── index.html              # Single-page browser UI (mic + chat)
│
├── logs/
│   └── calls.csv               # Persistent call log (auto-created)
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point: env checks → vector store → voice loop
│   ├── voice_loop.py           # Core voice conversation loop (mic mode)
│   ├── api.py                  # FastAPI backend for browser frontend
│   ├── agent.py                # Groq LLM calls + call metadata extraction
│   ├── rag.py                  # FAISS build / load / retrieve
│   ├── escalation.py           # EscalationManager: SLA timers + thresholds
│   ├── stt.py                  # Mic recording + Groq Whisper transcription
│   ├── tts.py                  # Edge TTS + interrupt detection via mic energy
│   └── logger.py               # CSV call logging
│
├── tests/
│   └── test_agent.py           # Unit tests for the agent module
│
├── Dockerfile                  # Docker image (port 7860 for HF Spaces)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)
- For voice mode: a working microphone and speakers
- For Docker deployment: Docker installed

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/agent-mcd.git
cd agent-mcd

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then open `.env` and set your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> The application performs a startup check and will exit with a clear error message if `GROQ_API_KEY` is missing or `data/solutions.txt` is not found.

### Running the Agent

**Option A — Voice Mode (microphone + speaker)**

```bash
python src/main.py
```

On first run, the FAISS vector store is built automatically from `data/solutions.txt`. Subsequent runs load the pre-built index from `data/faiss_index/`.

**Option B — Web API + Browser UI**

```bash
uvicorn src.api:app --reload --port 8000
```

Then open your browser at `http://localhost:8000`.

> ⚠️ **Do not open `frontend/index.html` directly as a file.** Chrome blocks microphone access on `file://` URLs. Always serve through the FastAPI backend.

---

## API Reference

The FastAPI backend exposes the following endpoints:

### `GET /health`
Returns `{"status": "ok"}`. Polled by the frontend every 8 seconds.

---

### `POST /session/start`
Creates a new support session and returns the opening greeting.

**Response:**
```json
{
  "session_id": "A1B2C3D4",
  "response": "Hello, thank you for calling McDonald's crew support..."
}
```

---

### `POST /session/message`
Sends a user message to the active session and receives Max's reply.

**Request:**
```json
{
  "session_id": "A1B2C3D4",
  "message": "Our POS terminal is completely frozen."
}
```

**Response:**
```json
{
  "response": "Please force close the POS application using Ctrl Alt Delete...",
  "priority": "P1",
  "failed_attempts": 0,
  "caller_name": "Sarah",
  "store_id": "341",
  "escalated": false,
  "resolved": false,
  "ended": false
}
```

---

### `POST /session/end`
Ends the session, logs the call, and cleans up server state.

**Request:**
```json
{ "session_id": "A1B2C3D4" }
```

---

### `POST /transcribe`
Accepts an audio file upload and returns Groq Whisper transcription.

**Form field:** `audio` (binary, `.webm` / `.wav` / `.mp3`)

**Response:**
```json
{ "text": "Our screen is not showing updated prices." }
```

---

## RAG Knowledge Base

The knowledge base lives in `data/solutions.txt` as structured plain-text blocks:

```
ISSUE: Screen not showing updated menu prices
PRIORITY: P2
SOLUTION: Go to the manager terminal and open the POS admin panel. Navigate to
Menu Management and click Sync Now. Wait 2 minutes for the sync to complete...
```

On startup, `rag.py` loads this file, splits it into 300-character chunks (50-character overlap) using `RecursiveCharacterTextSplitter`, embeds them with `all-MiniLM-L6-v2`, and saves the FAISS index to `data/faiss_index/`.

**To add new solutions**, append new `ISSUE / PRIORITY / SOLUTION` blocks to `solutions.txt` and delete `data/faiss_index/` to force a rebuild on the next run.

**To manually rebuild:**
```bash
python src/rag.py
```

---

## Priority & Escalation System

Priorities are auto-detected by matching the user's issue description against the RAG knowledge base and extracting the `PRIORITY:` tag from the retrieved chunks.

| Priority | Meaning | SLA |
|---|---|---|
| **P1** | Store down / unable to serve customers | 30 minutes |
| **P2** | Major system issue (POS, kiosk, display) | 60 minutes |
| **P3** | Partial issue (printer, login, portal) | 4 hours |
| **P4** | Minor issue (scheduling, employee records) | 24 hours |

**Escalation triggers:**

1. **Max attempts reached** — 3 failed solution attempts (user says "still not working", "already tried", etc.)
2. **SLA breach** — elapsed call time exceeds the priority's SLA limit
3. **Explicit user request** — caller says "escalate", "human agent", "supervisor", "transfer me", etc.
4. **Out-of-scope repeated** — caller makes 2+ messages unrelated to McDonald's IT/operations

**SLA warning:** When less than 20% of the SLA window remains, Max proactively informs the caller of the remaining time.

Priority can be **upgraded** (never downgraded) during the first two exchanges if the RAG context detects a more critical classification.

---

## Call Logging

Every call is appended to `logs/calls.csv` at the end of each session:

| Field | Description |
|---|---|
| `call_id` | Unique ID, e.g. `CALL-20240521143022` |
| `timestamp` | Date and time of the call |
| `caller_name` | Extracted via LLM from the caller's spoken response |
| `store_id` | Store number extracted via LLM (handles spoken digits like "3 4 1" → "341") |
| `issue_description` | One-sentence summary generated by LLM post-call |
| `priority` | P1–P4 |
| `resolution_status` | `Resolved` or `Unresolved` |
| `escalated_to_human` | `True` / `False` |
| `attempts` | Number of failed solution attempts |
| `duration_seconds` | Total call duration |

---

## Frontend

`frontend/index.html` is a single-file browser UI that:

- Connects to the FastAPI backend via `fetch`
- Shows a live chat transcript between the caller and Max
- Provides a **mic button** that records audio, posts it to `/transcribe`, and sends the transcription to `/session/message`
- Displays a sidebar with session metadata (priority, store ID, caller name, escalation status)
- Pings `/health` every 8 seconds to confirm backend connectivity

The frontend is served automatically at `http://localhost:8000` when you run `uvicorn src.api:app`.

---

## Docker & Deployment

### Build and run locally

```bash
docker build -t agent-mcd .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key agent-mcd
```

Open `http://localhost:7860` in your browser.

### Deploy to Hugging Face Spaces

The GitHub Actions workflow in `.github/workflows/deploy.yml` automatically syncs the `main` branch to your Hugging Face Space on every push.

**Setup steps:**

1. Create a [Hugging Face Space](https://huggingface.co/new-space) with **Docker** SDK.
2. Generate a Hugging Face write token at `huggingface.co/settings/tokens`.
3. Add `HF_TOKEN` as a GitHub repository secret (`Settings → Secrets → Actions`).
4. Add `GROQ_API_KEY` as a Space secret in your HF Space settings.
5. Update the `huggingface_repo_id` field in `deploy.yml` with your HF username and Space name.
6. Push to `main` — the pipeline builds, verifies, and deploys automatically.

---

## Running Tests

```bash
python tests/test_agent.py
```

The test file runs the agent through a simulated multi-turn conversation (frozen screen → restart didn't work → issue still persists) and prints the full exchange to stdout.

---

## How a Call Works

```
1. Caller connects
       │
       ▼
2. Max greets and asks for name
       │
       ▼
3. Caller states name → LLM extracts it
       │
       ▼
4. Max asks for store ID → LLM extracts it
       │
       ▼
5. Caller describes issue
       │
       ├──► Out-of-scope? → warn (×2 then close)
       │
       ├──► Priority detected from RAG (P1–P4), SLA timer starts
       │
       ├──► Explicit escalation request? → escalate immediately
       │
       ▼
6. RAG retrieves relevant solution steps
       │
       ▼
7. LLM generates one concrete step
       │
       ▼
8. Caller feedback:
       ├── "It worked" → mark Resolved, log call, end
       ├── "Still not working" → increment failed_attempts
       │       ├── attempts < 3 → go to step 7 with next RAG step
       │       └── attempts ≥ 3 → escalate to human
       └── SLA breached → escalate to human
       │
       ▼
9. Call logged to calls.csv
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key for LLM and Whisper |

All other configuration (model names, voice, SLA limits, log paths) is defined as constants inside the respective source modules and can be adjusted without environment variables.

---

## License

This project is for demonstration and educational purposes.

---

*Built with ❤️ using Groq, LangChain, FAISS, FastAPI, and Edge TTS.*
