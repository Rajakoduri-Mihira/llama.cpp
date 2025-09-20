# GPT-OSS Model Setup

This document describes how to download the GPT-OSS models (20B and 120B) in MXFP4 format for benchmarking.

## Download Instructions

### Prerequisites

Install Hugging Face CLI:
```bash
pip install huggingface_hub
```

Set your Hugging Face token (optional but recommended for faster downloads):
```bash
export HF_TOKEN=your_token_here
```

### GPT-OSS-20B (MXFP4)

```bash
# Download the 20B model
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-mxfp4.gguf --local-dir ./models/ggml-org --token $HF_TOKEN
```

### GPT-OSS-120B (MXFP4)

```bash
# Download the 120B model (split into 3 parts)
hf download ggml-org/gpt-oss-120b-GGUF --include "gpt-oss-120b-mxfp4-*.gguf" --local-dir ./models/ggml-org --token $HF_TOKEN
```

## Model Information

### GPT-OSS-20B
- **Model**: GPT-OSS-20B MXFP4 MoE
- **Size**: 11.27 GiB
- **Parameters**: 20.91B
- **Quantization**: MXFP4 (native)
- **Source**: [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)

### GPT-OSS-120B
- **Model**: GPT-OSS-120B MXFP4 MoE
- **Size**: ~75 GiB (split into 3 files)
- **Parameters**: 120.7B
- **Quantization**: MXFP4 (native)
- **Source**: [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)

## Benchmark Results

### RTX 5090 Performance (20B Model)

| Test Configuration | Performance | Description |
|-------------------|-------------|-------------|
| **pp512** (Prompt Processing) | ~10,150 t/s | Processing 512 prompt tokens |
| **pp1024** (Prompt Processing) | ~9,550 t/s | Processing 1024 prompt tokens |
| **tg128** (Text Generation) | ~318 t/s | Generating 128 tokens |
| **tg512** (Text Generation) | ~300 t/s | Generating 512 tokens |

## Usage

### Individual Benchmarks

```bash
# Run 20B model benchmark
./benchmark-single-gpu.sh 0 gpt20b

# Run 120B model benchmark (requires sufficient VRAM)
./benchmark-single-gpu.sh 0 gpt120b

# Run all available benchmarks
./benchmark-single-gpu.sh 0 all
```

### Manual llama-bench Usage

```bash
# 20B model
./build/bin/llama-bench -m models/ggml-org/gpt-oss-20b-mxfp4.gguf

# 120B model
./build/bin/llama-bench -m models/ggml-org/gpt-oss-120b-mxfp4-00001-of-00003.gguf
```

## Expected File Paths

After download, models should be located at:
- **20B**: `models/ggml-org/gpt-oss-20b-mxfp4.gguf`
- **120B**: `models/ggml-org/gpt-oss-120b-mxfp4-*.gguf` (3 files)

## Hardware Requirements

- **20B Model**: ~12GB VRAM minimum
- **120B Model**: ~32GB+ VRAM minimum (may require model sharding for smaller GPUs)

## Notes

- MXFP4 quantization provides excellent performance with minimal quality loss
- The 120B model may not fit on single consumer GPUs without additional optimization
- Both models use native MXFP4 precision for MoE layers, enabling efficient inference