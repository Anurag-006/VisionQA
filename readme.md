# VisionQA

**A fully offline, on-device visual question-answering assistant for Android.**

Point your camera at anything — a menu, a sign, a document, an object — and ask questions about it in natural language or by voice. No network calls, no cloud inference, no data leaving the device.

---

## Why this exists

Most "ask your camera a question" apps ship the image to a server. VisionQA doesn't — the entire pipeline, from OCR to language understanding to response generation, runs locally on the phone. That constraint shaped every architectural decision in this project: model size, memory management, and latency all had to be solved on-device, not offloaded to a GPU cluster.

---
## Architecture

```text
                    ┌──────────────┐
                    │   CameraX    │
                    │   Capture    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Bitmap    │
                    │   Capture    │
                    └──────┬───────┘
                           │
                           ▼
              ┌─────────────────────────────┐
              │       OCR Extraction        │
              │ (ML Kit + Spatial Layout)   │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │        Query Router         │
              │     (OcrQueryHandler)       │
              └─────────────┬───────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
        ┌──────────────────────┐ ┌──────────────────────┐
        │    Structured OCR    │ │    MediaPipe LLM     │
        │    Direct Answer     │ │   Inference Engine   │
        │                      │ │    (Gemma 3n E2B)    │
        └──────────┬───────────┘ └──────────┬───────────┘
                   │                        │
                   └──────────┬─────────────┘
                              ▼
              ┌─────────────────────────────┐
              │    Streaming Response       │
              │     Text + Optional TTS     │
              └─────────────────────────────┘
```


## Pipeline stages

**1. Capture** — `CameraController` wraps CameraX's `ImageCapture` use case to grab a frame on demand.

**2. OCR with spatial reconstruction** — `OcrHelper` wraps ML Kit's text recognizer, but doesn't stop at raw text. Most OCR wrappers just concatenate `result.text` and lose all layout information. This one reconstructs structure from bounding-box geometry:
- Clusters text elements into logical rows using a **dynamic y-threshold** derived from average line height (not a fixed pixel value — so it works across image resolutions)
- Detects the true column split point by finding the gap between left- and right-column elements across multiple rows, rather than naively splitting at the image midpoint
- Outputs `StructuredLine` objects pairing left/right text (e.g. `"Biryani"` ↔ `"Rs. 250/-"`), correctly handling menus, price lists, and forms regardless of how ML Kit orders the raw text

**3. Query routing** — `OcrQueryHandler` classifies each question before deciding whether the LLM needs to run at all:
- Price/list/comparison questions ("what's the cheapest item?") are answered **directly from structured OCR data** — no model inference required
- General/open-ended questions fall through to the vision-language model

This routing is a deliberate latency and battery optimization: a large chunk of real-world questions about a photographed menu or sign are structurally answerable without ever invoking the LLM.

**4. On-device vision-language inference** — `GemmaVLM` runs **Google's MediaPipe LLM Inference API** with a quantized **Gemma 3n E2B** model (~3.1GB, int4). Key details:
- Runs fully offline after a one-time model download (`GemmaDownloader`, with resumable progress reporting and integrity checks)
- Images are capped to a max dimension before encoding to control memory pressure
- Each new photo starts a fresh `LlmInferenceSession`; `resetForNewImage()` explicitly tears down and reinitializes session state to avoid cross-image context bleed
- Responses stream token-by-token via a coroutine-based callback (`generateResponseAsync`), so the UI shows text as it's generated rather than waiting for a full response

**5. Voice I/O** — `SpeechManager` wraps Android's native `SpeechRecognizer` and `TextToSpeech` for hands-free question-asking and spoken answers, using a `Flow`-based callback bridge for streaming partial transcription results.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Kotlin, Jetpack Compose |
| Camera | CameraX |
| OCR | Google ML Kit Text Recognition |
| VLM inference | MediaPipe LLM Inference API (Gemma 3n E2B, int4 quantized) |
| Voice | Android SpeechRecognizer / TextToSpeech |
| Async | Kotlin Coroutines, Flow |
| Networking | OkHttp (model download) |

---

## What makes this interesting from an engineering standpoint

- **No naive OCR-to-text pipeline.** The spatial reconstruction algorithm solves a real problem (layout-aware text extraction) that a plain `result.text` call cannot.
- **Compute-aware routing.** Not every question needs a 2B-parameter model — the router recognizes when structured data alone can answer it, saving latency and battery.
- **Explicit session lifecycle management.** Rather than letting the LLM session accumulate stale context across images, session teardown/reinit is handled explicitly on every new capture.
- **Fully offline inference on commodity Android hardware** — no server, no API key required at runtime (only for the one-time model download).

---

## Setup

1. Clone the repo
2. Set up model download credentials in `local.properties` (do **not** hardcode tokens in source — see [Configuration](#configuration))
3. Build via Android Studio or `./gradlew assembleDebug`
4. On first launch, the app downloads the Gemma 3n E2B model (~3.1GB) — Wi-Fi recommended

### Configuration

The model download requires a Hugging Face access token. Set it as a Gradle property or environment variable rather than hardcoding it in `GemmaDownloader.kt`:

```properties
# local.properties (not committed to VCS)
HF_TOKEN=your_token_here
```

---

## Roadmap / possible extensions

- Re-introduce a fully custom on-device inference path (llama.cpp + multimodal projector) as an alternative backend, enabling manual KV-cache control for advanced use cases
- Benchmark suite for p50/p95 latency and peak RAM across device tiers
- On-device caching of OCR results per image to avoid re-running recognition on retry