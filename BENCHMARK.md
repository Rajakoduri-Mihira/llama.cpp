# llama.cpp Benchmarking Guide

This guide provides comprehensive instructions for benchmarking and running llama.cpp models on single and multi-GPU configurations.

## Available Models

| Model | Size | Parameters | Format | File |
|-------|------|------------|--------|------|
| **Gemma 3-1B** | 769MB | 3.1B | Q4_K_M | `gemma-3-1b-it-Q4_K_M.gguf` |
| **GPT-OSS 20B** | 11.3GB | 20.9B | MXFP4 | `gpt-oss-20b-mxfp4.gguf` |
| **GPT-OSS 120B** | 59GB | 116.8B | MXFP4 | `gpt-oss-120b-mxfp4-00001-of-00003.gguf` |

## Quick Start

### Download Models

```bash
# Download models using HuggingFace (requires HF_TOKEN for some models)
export HF_TOKEN=your_token_here

# Gemma 3B
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "test" -n 1

# GPT-OSS 20B
./build/bin/llama-cli -hf ggml-org/gpt-oss-20b-GGUF -p "test" -n 1

# GPT-OSS 120B (large download ~88GB)
./build/bin/llama-cli -hf ggml-org/gpt-oss-120b-GGUF -p "test" -n 1
```

## Benchmark Scripts

### 1. Single GPU Benchmark

Benchmarks models on a single GPU.

```bash
./benchmark-single-gpu.sh [GPU_ID] [MODEL]
```

**Parameters:**
- `GPU_ID`: GPU index (0-based), default: 0
- `MODEL`: `gemma`, `gpt20b`, `gpt120b`, or `all` (default)

**Examples:**
```bash
# Benchmark all models on GPU 0
./benchmark-single-gpu.sh

# Benchmark Gemma on GPU 2
./benchmark-single-gpu.sh 2 gemma

# Benchmark GPT-20B on GPU 0
./benchmark-single-gpu.sh 0 gpt20b
```

### 2. Multi-GPU Benchmark

Benchmarks models across multiple GPUs.

```bash
./benchmark-multi-gpu.sh [GPU_LIST] [MODEL]
```

**Parameters:**
- `GPU_LIST`: Comma-separated GPU IDs or "all" (default)
- `MODEL`: `gemma`, `gpt20b`, `gpt120b`, or `all` (default)

**Examples:**
```bash
# Benchmark all models on all GPUs
./benchmark-multi-gpu.sh

# Benchmark GPT-120B on GPUs 0,1,2,3
./benchmark-multi-gpu.sh "0,1,2,3" gpt120b

# Benchmark GPT-20B on all available GPUs
./benchmark-multi-gpu.sh all gpt20b
```

### 3. Complete Benchmark Suite

Runs comprehensive benchmarks comparing single vs multi-GPU performance.

```bash
./benchmark-suite.sh [OUTPUT_FILE]
```

**Features:**
- Tests each model on single GPU, 2 GPUs, 4 GPUs (if available), and all GPUs
- Saves results to timestamped file
- Generates performance comparison table

**Example:**
```bash
# Run complete suite with default output file
./benchmark-suite.sh

# Save to specific file
./benchmark-suite.sh results_amd_7900xtx.txt
```

### 4. Interactive Model Launcher

Launch models for interactive use, API server, or CLI testing.

```bash
./launch-model.sh [MODEL] [GPU_LIST] [MODE]
```

**Parameters:**
- `MODEL`: `gemma`, `gpt20b`, or `gpt120b`
- `GPU_LIST`: Single GPU ID, comma-separated list, or "all"
- `MODE`: `cli` (default), `interactive`, or `server`

**Examples:**
```bash
# Interactive chat with Gemma on GPU 0
./launch-model.sh gemma 0 interactive

# Launch GPT-20B API server on GPUs 0,1
./launch-model.sh gpt20b "0,1" server

# CLI mode with GPT-120B on all GPUs
./launch-model.sh gpt120b all cli
```

**Server Mode Ports:**
- Gemma: `http://localhost:8080`
- GPT-20B: `http://localhost:8081`
- GPT-120B: `http://localhost:8082`

## Performance Expectations

### AMD RX 7900 XTX (Single GPU)

| Model | Prompt Processing | Text Generation | VRAM Usage |
|-------|------------------|-----------------|------------|
| **Gemma 3-1B** | 19,051 t/s | 238 t/s | ~2GB |
| **GPT-OSS 20B** | 3,294 t/s | 140 t/s | ~11GB |
| **GPT-OSS 120B** | OOM | OOM | >24GB |

### AMD RX 7900 XTX (Multi-GPU)

| Model | Configuration | Prompt Processing | Text Generation | VRAM Usage |
|-------|---------------|------------------|-----------------|------------|
| **Gemma 3-1B** | 2 GPUs | 15,786 t/s* | 196 t/s* | ~2GB |
| **Gemma 3-1B** | 6 GPUs | 15,775 t/s* | 142 t/s* | ~2GB |
| **GPT-OSS 20B** | 2 GPUs | 3,033 t/s* | 115 t/s* | ~11GB |
| **GPT-OSS 20B** | 6 GPUs | 2,535 t/s* | 97 t/s* | ~11GB |
| **GPT-OSS 120B** | 4 GPUs | 1,448 t/s | 75 t/s | ~60GB |
| **GPT-OSS 120B** | 6 GPUs | 1,391 t/s | 68 t/s | ~60GB |

*Multi-GPU overhead reduces performance for smaller models due to communication latency

## Optimal GPU Configuration

| Model | Recommended Setup | Reason |
|-------|------------------|--------|
| **Gemma 3-1B** | Single GPU | Model fits easily, multi-GPU adds overhead |
| **GPT-OSS 20B** | Single GPU | Fits on 24GB VRAM, minimal benefit from multi-GPU |
| **GPT-OSS 120B** | 4-6 GPUs | Requires >24GB VRAM, benefits from distribution |

## Advanced Usage

### Custom Benchmark Parameters

Edit the scripts to modify:
- `PROMPT_SIZE`: Token count for prompt processing (default: 512)
- `GEN_SIZE`: Token count for generation (default: 128)
- `REPETITIONS`: Number of benchmark runs (default: 3)

### Environment Variables

```bash
# AMD GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3

# NVIDIA GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Force specific backend
export GGML_CUDA_NO_PEER_COPY=1  # Disable peer-to-peer for stability
```

### Memory Management

For large models on multi-GPU:
```bash
# Check GPU memory
rocm-smi --showmeminfo vram  # AMD
nvidia-smi                   # NVIDIA

# Monitor during runtime
watch -n 1 rocm-smi          # AMD
watch -n 1 nvidia-smi        # NVIDIA
```

## Troubleshooting

### Out of Memory (OOM)

**Problem:** Model fails to load with OOM error

**Solutions:**
1. Reduce context size: Add `-c 2048` to commands
2. Use fewer GPU layers: `-ngl 50` instead of `-ngl 99`
3. Use more GPUs: Distribute model across multiple cards
4. Use smaller quantization: Try Q4_0 instead of Q4_K_M

### Slow Multi-GPU Performance

**Problem:** Multi-GPU slower than single GPU for small models

**Solution:** This is expected due to communication overhead. Use single GPU for models under 20B parameters.

### GPU Not Detected

**Problem:** Scripts don't detect GPUs

**Check:**
```bash
# AMD
rocm-smi --showbus

# NVIDIA
nvidia-smi --list-gpus
```

### Model Download Issues

**Problem:** Model download fails or is slow

**Solutions:**
1. Use a HuggingFace token for gated models
2. Download directly from HuggingFace website
3. Use `wget` or `curl` with resume support:
```bash
wget -c https://huggingface.co/ggml-org/model-name/resolve/main/file.gguf
```

## Performance Metrics Explained

- **pp (Prompt Processing):** Tokens processed per second during initial prompt evaluation
- **tg (Text Generation):** Tokens generated per second during inference
- **Time to First Token (TTFT):** Latency before first token appears (≈1000/tg ms)
- **t/s:** Tokens per second

## MXFP4 Format Details

The MXFP4 format used by GPT-OSS models:
- **4-bit quantization** with custom encoding
- **On-the-fly GPU decompression** using lookup tables
- **Memory efficient:** 4x smaller than FP16
- **Performance optimized:** Uses AMD GCN intrinsics on ROCm

## Build Requirements

Ensure llama.cpp is built with appropriate backend:

```bash
# AMD HIP/ROCm
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release -j $(nproc)

# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# CPU only (fallback)
cmake -B build
cmake --build build --config Release -j $(nproc)
```

# Performance Results

## Hardware Configuration

- **GPU Setup**: 6x AMD Radeon RX 7900 XTX (24GB VRAM each)
- **Backend**: ROCm/HIP
- **System**: AMD64 Linux
- **Build**: llama.cpp with HIP backend (`-DGGML_HIP=ON`)
- **Test Configuration**: 512 token prompts, 128 token generation, 3 repetitions average

## Actual Benchmark Results

### Comprehensive Performance Matrix

| Model | Configuration | Prompt Processing | Text Generation | VRAM Distribution |
|-------|---------------|------------------|-----------------|------------------|
| **Gemma 3-1B (Q4_K_M)** | Single GPU (0) | 19,051.28 t/s | 237.65 t/s | GPU0: ~2GB |
| **Gemma 3-1B (Q4_K_M)** | 2 GPUs (0,1) | 15,785.85 t/s | 196.18 t/s | Split across 2 GPUs |
| **Gemma 3-1B (Q4_K_M)** | All 6 GPUs | 15,774.84 t/s | 141.94 t/s | Split across 6 GPUs |
| **GPT-OSS 20B (MXFP4)** | Single GPU (0) | 3,294.36 t/s | 140.36 t/s | GPU0: ~11GB |
| **GPT-OSS 20B (MXFP4)** | 2 GPUs (0,1) | 3,032.76 t/s | 114.90 t/s | Split across 2 GPUs |
| **GPT-OSS 20B (MXFP4)** | All 6 GPUs | 2,534.73 t/s | 97.44 t/s | Split across 6 GPUs |
| **GPT-OSS 120B (MXFP4)** | Single GPU | OOM | OOM | >24GB required |
| **GPT-OSS 120B (MXFP4)** | 4 GPUs | 1,447.98 t/s | 75.36 t/s | ~15GB per GPU |
| **GPT-OSS 120B (MXFP4)** | All 6 GPUs | 1,390.61 t/s | 68.19 t/s | ~10GB per GPU |

### Key Performance Insights

1. **Single GPU Optimal**: Smaller models (≤20B) perform best on single GPU due to zero communication overhead
2. **Multi-GPU Scaling**: Only beneficial for models >24GB VRAM requirement (GPT-OSS 120B)
3. **MXFP4 Efficiency**: Excellent 4.34 BPW compression with native HIP acceleration via lookup tables
4. **Memory Distribution**: 120B model requires 59GB total, distributes automatically across available GPUs
5. **Time to First Token**: Sub-millisecond latency across all configurations
6. **Model Loading**: 120B model loads in ~19 seconds with complete GPU offloading (37/37 layers)

### GPU Memory Allocation (120B Model, 6 GPUs)

- ROCm0: 11,523.72 MiB (primary/output layer)
- ROCm1-4: 9,877.47 MiB each (transformer layers)
- ROCm5: 8,818.06 MiB (remaining layers)
- CPU_Mapped: 586.82 MiB (metadata/embeddings)

## Additional Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Model Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Contributing

When benchmarking new models or configurations:
1. Run `./benchmark-suite.sh` for comprehensive results
2. Save output with descriptive filename
3. Include GPU model, driver version, and ROCm/CUDA version
4. Note any special settings or modifications

## License

These benchmark scripts are provided as-is for testing llama.cpp performance.
Refer to llama.cpp repository for model and software licenses.