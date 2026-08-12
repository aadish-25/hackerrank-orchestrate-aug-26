# WhatsApp Message Notification Router
### HackerRank Orchestrate — August 2026

An AI-powered system that classifies every incoming WhatsApp message into one of three routing actions: **notify**, **digest**, or **mute** — personalised to the receiving user. The system handles multimodal messages (text, image, voice note), defends against prompt injection and domain fraud, and is built for resilience with full checkpoint-based resumption and a three-tier LLM fallback chain.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp ../.env.example ../.env
# Edit .env and fill in GEMINI_API_KEY and GROQ_API_KEY

# 3. Run the router on messages.csv → output.csv
python main.py

# 4. Evaluate against sample_messages.csv ground truth
python main.py --evaluate
```

---

## Architecture

The system is built on a **model-observes / code-decides** pattern: the LLM is responsible only for reading and understanding the message, while all safety-critical overrides happen in deterministic Python code that the LLM cannot influence.

### Pipeline Graph (LangGraph StateGraph)

```
context_node → media_node → routing_node → gate_node
```

Each node is a separate Python module with a single, well-scoped responsibility:

| Node | Module | Responsibility |
|---|---|---|
| `context_node` | `pipeline/context.py` | Loads all 10 dataset CSVs into a single context dict for the message being processed. Detects DND window, domain fraud, user mentions, group mute state. |
| `media_node` | `pipeline/extraction.py` | Processes image (OCR via Gemini Vision) and voice (transcript via Groq Whisper) messages. Extracts structured `MediaExtraction` Pydantic object with urgency signals, URL detection, OTP detection, and prompt injection flags. Results are cached per `media_id`. |
| `routing_node` | `pipeline/routing.py` | Builds a rich, context-aware prompt and calls the LLM fallback chain. Returns a structured `RoutingDecision` Pydantic object. |
| `gate_node` | `pipeline/gate.py` | Applies 7 deterministic override rules in pure Python. Zero LLM calls. Can override or downgrade the model's decision. |

After the graph runs, two final modules compute the output:
- `pipeline/evidence.py`: Direction-sensitive evidence scoring to select the top 3 historical messages.
- `pipeline/confidence.py`: Weighted heuristic confidence calculation (avoids LLM self-rating bias).

---

## Key Architectural Decisions

### 1. Model Observes, Code Decides
The LLM's role is to read and understand message content. All hard safety rules — injection detection, domain fraud, viral forward blocking, DND enforcement, opt-out enforcement — are executed in deterministic Python after the LLM returns its answer. This means:
- The LLM cannot be manipulated into bypassing a rule.
- Safety rules are testable independently of the LLM.
- Gate overrides are transparently labeled in the `reason` field of `output.csv`.

### 2. Two-Layer Prompt Injection Defense
Injection defense is OR'd across two independent layers so neither layer needs to be exhaustive alone:
- **Layer 1 (LLM):** The routing LLM self-reports a `potential_prompt_injection` flag and a `injection_confidence` score as part of its structured output. This catches paraphrased or translated injection attempts.
- **Layer 2 (Regex):** A tight regex backstop in `gate.py` matches literal injection patterns (`set action=`, `classify as notify`, `system note for the notification router`, etc.). This catches literal injections even if the LLM missed them.

### 3. Three-Tier LLM Fallback Chain (Transparent)
Rather than using LangChain's opaque `.with_fallbacks()` which silently swallows errors, we implement an explicit manual iteration loop. This gives full control over when and how fallbacks are triggered:
- **Primary:** Gemini 3.5 Flash
- **Secondary:** Groq Llama 3.3 70B  
- **Tertiary:** Local Qwen2.5 7B Instruct (via LM Studio, if running)

Each request prints the active provider to the console (e.g., `[Gemini]`, `[Groq]`), making every run fully observable for debugging. The active provider is confirmed after each successful response. This is a deliberate deviation from LangChain's built-in `.with_fallbacks()`, which was avoided precisely because it offers no visibility into which provider actually handled the request — an unacceptable blind spot when debugging live API quota behaviour.

### 4. Token Bucket Rate Limiter + Hard Quota Exhaustion Handler
A custom `RateLimiter` class (Token Bucket algorithm using a `collections.deque`) is applied only to the primary Gemini model, enforcing a maximum of 14 requests per 60-second window — safely within Gemini's free-tier limit of 15 RPM. This is a deliberate deviation from using LangChain's built-in `.with_retry()`, chosen so that rate-limit pacing applies exclusively to the primary provider rather than uniformly across all fallbacks, and so the wait duration is visible in the console rather than silently handled inside the framework.

If Gemini returns a `429 RESOURCE_EXHAUSTED` error, the pipeline distinguishes between two types of quota limits:
- **Per-minute burst limit:** Waits 60 seconds and retries Gemini. If it succeeds, continues using Gemini.
- **Daily quota exhaustion:** If the retry also fails, the pipeline hard-stops with a clear message: `"Please swap your GEMINI_API_KEY and run again — it will resume from where it left off."` This prevents the pipeline from silently degrading to a weaker model for the primary dataset.

### 5. Segmented, Per-Record Checkpoint System
Checkpoints are saved after every single record and split across three separate files:

| File | Contains | When to clear |
|---|---|---|
| `cache/checkpoint_media.json` | OCR + Whisper media extraction results | Only when media inputs change |
| `cache/checkpoint_routing.json` | LLM routing decisions per `message_id` | When tuning the prompt or routing logic |
| `cache/checkpoint_eval.json` | Evaluation run results | When re-evaluating against ground truth |

Segmentation is crucial: media extraction (OCR, audio transcription) is slow and expensive. Clearing routing checkpoints to re-run the LLM with a new prompt does not re-process media. A single crashed record never corrupts the batch — the runner skips already-completed records on restart.

### 6. Direction-Sensitive Evidence Selection
Before the LLM is called, the evidence module (`pipeline/evidence.py`) scores historical messages using two separate weight vectors depending on which direction the current message leans:
- **MUTE signals:** `notification_dismissed (0.35)`, `muted_after_message (0.35)`, `message_reported (0.30)`
- **NOTIFY signals:** `message_replied (0.40)`, `fast_reaction_time (0.35)`, `message_opened (0.25)`

Only the top 3 most relevant historical messages are injected into the prompt. This prevents the LLM from drowning in irrelevant history and ensures evidence is directionally aligned with the routing decision being made.

### 7. Code-Computed Confidence (No LLM Self-Rating)
Confidence scores are computed entirely in Python from objective signals, weighted by `CONFIDENCE_WEIGHTS` in `config.py`:

| Signal | Weight |
|---|---|
| Business account verified | 0.20 |
| Prior engagement positive | 0.20 |
| User opted in | 0.15 |
| No injection flag detected | 0.15 |
| Sender domain matches brand | 0.15 |
| Sender trust (known user) | 0.15 |

Gate overrides are assigned a fixed confidence of `0.92` — high, but deliberately not `1.0`, because a hardcoded rule can still misjudge an edge case that was not anticipated when the rule was written (for example, a legitimate same-day school notice arriving during DND). Setting it at `0.92` signals strong but not absolute certainty, which is more honest and better calibrated than claiming infallibility.

---

## Deterministic Safety Gate Rules

Applied in priority order. First matching rule wins and overrides the LLM.

| Priority | Rule | Action |
|---|---|---|
| 1 | **Injection defense** — Layer 1 (LLM flag) OR Layer 2 (regex match) | `mute / scam` |
| 2 | **Domain fraud** — sender domain is newly registered or mismatches brand | `mute / scam` |
| 3 | **Unverified business + payment/OTP request** in media | `mute / scam` |
| 4 | **Viral forward** — `forwarded_count >= 8` | `mute / forward` |
| 5 | **Promotion opt-out enforced** — user explicitly opted out from this business | `mute / promotion` |
| 6 | **Muted group, no mention** — `notify` downgraded if group muted and user not @mentioned | `digest` |
| 7 | **DND window** — `notify` downgraded if message arrives in user's quiet hours | `digest` |

---

## CLI Reference

All commands run from the `code/` directory.

```bash
# Default run — skips completed records, retries errors
python main.py

# Dry-run — context loading + safety gate only, zero API calls ($0 cost)
python main.py --dry-run

# Evaluate against sample_messages.csv ground truth
python main.py --evaluate

# Re-run gate + confidence over cached LLM outputs ($0 cost, fast)
python main.py --recompute-gate

# Clear routing cache only, keep expensive media cache intact
python main.py --force-routing

# Clear media cache only
python main.py --force-media

# Wipe everything and restart from scratch
python main.py --force-clean

# Combine flags — e.g., fresh routing eval, skip confirmation prompt
python main.py --evaluate --force-routing -y
```

---

## Project Structure

```
code/
├── main.py                    # CLI entry point, orchestrates the full pipeline
├── config.py                  # All paths, model names, weights, and thresholds
├── schemas.py                 # Pydantic models: RoutingDecision, MediaExtraction, FinalOutputRow
├── requirements.txt           # Pinned dependencies
├── README.md                  # This file
├── pipeline/
│   ├── context.py             # Loads all dataset CSVs into a per-message context dict
│   ├── extraction.py          # Multimodal media extraction (OCR + Whisper)
│   ├── evidence.py            # Direction-sensitive historical evidence scoring
│   ├── routing.py             # LLM fallback chain, rate limiter, prompt builder
│   ├── gate.py                # Deterministic safety gate (7 override rules)
│   ├── confidence.py          # Code-computed confidence scoring
│   └── graph.py               # LangGraph StateGraph assembly
├── checkpoints/
│   └── manager.py             # Per-record checkpoint read/write/skip logic
└── evaluation/
    ├── main.py                # Evaluation runner (scoped to sample_messages.csv)
    └── scorer.py              # Field-by-field scorer with action, type, evidence metrics
```

---

## Setup & Requirements

**Python:** 3.10+

**Install:**
```bash
pip install -r requirements.txt
```

**Environment variables** (create a `.env` file in the repo root):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides
PRIMARY_MODEL=gemini-3.5-flash
FALLBACK_MODEL=llama-3.3-70b-versatile
LOCAL_MODEL_URL=http://localhost:1234/v1
```

A `.env.example` template is included at the repo root. Secrets are never hardcoded — all API keys are loaded exclusively from environment variables via `python-dotenv`.

---

## Evaluation Results (sample_messages.csv)

Achieved on a 30-record holdout from `sample_messages.csv` using Gemini 3.5 Flash as the primary model:

| Metric | Score |
|---|---|
| Action Accuracy | **80.0%** (24/30) |
| Message Type Accuracy | **56.7%** (17/30) |
| Evidence Relevance | **53.3%** (16/30) |

Prompt injection detection: **5/5 caught** by the safety gate.  
Domain fraud detection: **6/6 caught** by the safety gate.

Message type accuracy (56.7%) is the weakest of the three metrics. The most likely cause is boundary confusion between adjacent categories — the model occasionally conflates `spam` with `promotion`, or classifies a personal-but-urgent message as `personal` rather than `urgent`. The clearest next improvement would be adding tighter boundary examples for these adjacent pairs directly into the routing prompt, so the LLM has explicit guidance on where one category ends and the other begins.
