# Emotional AI Agent

A privacy-preserving emotional support chatbot that runs **entirely on your device** — no cloud, no data leaving your machine.

It detects the emotion in your messages in real time and adapts its tone accordingly, powered by a local LLM served through [Ollama](https://ollama.com).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| [Ollama](https://ollama.com/download) | latest | Runs the local LLM |
| RAM | 4 GB minimum | See model guide below |

### Install Ollama

**macOS / Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

Download and run the installer from [ollama.com/download](https://ollama.com/download).

Verify the install:
```bash
ollama --version
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd emotional-ai-agent

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Train the emotion model (one-time, ~5 minutes)
python3 scripts/train_emotion_model.py
```

---

## Running the app

```bash
./run.sh
```

That's it. The script will:

1. Check that Ollama is installed
2. Start `ollama serve` in the background if it isn't already running
3. Pull the default model if you don't have it yet (first run only)
4. Launch the Streamlit interface in your browser

On first launch you'll be asked for your name and how much RAM you have — the app uses that to pick the best model for your machine.

> **Windows users:** run the equivalent commands manually:
> ```bat
> ollama serve
> streamlit run app/main.py
> ```

---

## Model guide

The dropdown in the top bar shows each model's **total RAM footprint** — LLM weights plus the emotion detection model loaded alongside it. The app picks a default based on the RAM you enter at first launch, but you can switch at any time. When you switch, the previous model is automatically unloaded before the new one loads.

### RAM sizes at a glance

| Model | LLM weights | + Emotion model | **Total in RAM** |
|---|---|---|---|
| Gemma 2 2B (`gemma2:2b-instruct-q4_K_M`) | 1.7 GB | + 0.1–0.3 GB | **~1.8–2.0 GB** |
| Phi-3 Mini (`phi3:mini`) | 2.2 GB | + 0.3 GB | **~2.5 GB** |
| Mistral 7B (`mistral`) | 4.1 GB | + 0.3 GB | **~4.4 GB** |
| Qwen 2.5 14B (`qwen2.5:14b`) | 9.0 GB | + 0.3 GB | **~9.3 GB** |

> The emotion model column shows 0.1 GB for MiniLM (selected automatically on 4 GB machines) and 0.3 GB for DistilBERT (used on 8 GB and above).

### What to pick for your machine

| Your RAM | Best choice | Why |
|---|---|---|
| 4 GB | Gemma 2 2B | Only model that leaves enough headroom for macOS/Windows |
| 6–8 GB | Gemma 2 2B or Phi-3 Mini | Phi-3 Mini fits comfortably; gives noticeably better responses |
| 12–16 GB | Mistral 7B | Good balance of quality and speed; ~4.4 GB leaves plenty free |
| 24–32 GB | Mistral 7B or Qwen 2.5 14B | Qwen gives the best quality if you have the headroom |
| 32 GB+ | Qwen 2.5 14B | Runs with headroom to spare |

**Rule of thumb:** your machine needs roughly 2× the model's RAM footprint free before launching, so macOS or Windows can keep running smoothly alongside it.

To pull any model manually before first run:
```bash
ollama pull mistral          # 4.1 GB download
ollama pull phi3:mini        # 2.2 GB download
ollama pull qwen2.5:14b      # 9.0 GB download
```

---

## Privacy

- **Standard mode** — conversations are saved to `data/conversations.db` with AES-256 encryption. Data never leaves your device.
- **Privacy mode** — nothing is written to disk. All data lives in memory only and is gone when you close the app.

Toggle Privacy Mode in the ⚙ Settings popover at any time.

---

## Project structure

```
app/
  main.py               Streamlit UI
  components/           UI widgets (emotion badge, etc.)
config/
  settings.py           All configuration, reads from .env
src/
  emotion/              Emotion detection model
  smoothing/            Context-window emotion smoother
  llm/                  Ollama HTTP client + prompt builder
  pipeline/             EmotionalAgent orchestrator
  storage/              Encrypted SQLite storage
scripts/
  train_emotion_model.py  One-time model training
data/
  conversations.db      Encrypted conversation history
run.sh                  Start script
```

---

## Troubleshooting

**"Ollama is not running"**
Run `ollama serve` in a separate terminal, or just use `./run.sh` which handles this automatically.

**"Model not found"**
Pull the model: `ollama pull <model-name>`. The run script does this automatically for the default model.

**Slow first response**
The model is being loaded into RAM. Subsequent messages in the same session are fast. The run script pre-loads the model during startup to minimise this.

**"Emotion model not trained"**
Run `python3 scripts/train_emotion_model.py` once before starting the app.
