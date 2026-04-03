![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Video Generation Blueprint</h1>

<p align="center"><em>Decentralized video generation on <a href="https://tangle.tools">Tangle</a> — operators serve Hunyuan Video, LTX-Video, and other video models via ComfyUI or API.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to serve video generation with anonymous payments through shielded credits. Supports local GPU inference (ComfyUI workflows) or API-based backends (Modal, Replicate). Async job model — generation takes 30s-5min.

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable results on Tangle
- **x402 HTTP** — private video generation at `/v1/video/generate`

Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) with TEE support.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — ComfyUI/API dual mode, async job store, SpendAuth billing |
| `contracts/` | Solidity | VideoGenBSM — VRAM validation, per-second pricing, duration limits |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/video/generate` | Submit video generation job |
| GET | `/v1/video/:job_id` | Poll job status / download result |
| GET | `/health` | Operator health check |
| GET | `/v1/operator` | Capabilities, pricing, active jobs |

## Backends

- **Local mode**: ComfyUI with Hunyuan Video / LTX-Video workflows. Requires 48GB+ VRAM.
- **API mode**: Forwards to Modal or Replicate. No local GPU required.

## Pricing

Per-second-of-video. Configured by Blueprint admin via `configureModel()`. High VRAM requirements enforce quality operators.

## TEE Support

Add `features = ["tee"]` to `blueprint-sdk` in Cargo.toml. The `TeeLayer` middleware transparently attaches attestation metadata when running in a Confidential VM. Passes through when no TEE is configured.

## Quick Start

```bash
# Local mode (requires ComfyUI + Hunyuan Video)
VIDEO_ENDPOINT=http://localhost:8188 VIDEO_MODE=local cargo run --release

# API mode (requires Modal/Replicate API key)
VIDEO_ENDPOINT=https://api.replicate.com VIDEO_MODE=api cargo run --release
```

## Related Repos

- [Blueprint SDK](https://github.com/tangle-network/blueprint) — framework for building Blueprints
- [vLLM Inference Blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — text inference
- [Voice Inference Blueprint](https://github.com/tangle-network/voice-inference-blueprint) — TTS/STT
- [Image Generation Blueprint](https://github.com/tangle-network/image-gen-inference-blueprint) — image generation
- [Embedding Blueprint](https://github.com/tangle-network/embedding-inference-blueprint) — text embeddings
