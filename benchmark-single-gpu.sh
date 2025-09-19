#!/bin/bash
# Single GPU Benchmark Script for llama.cpp models
# Usage: ./benchmark-single-gpu.sh [GPU_ID] [MODEL]
# Models: gemma, gpt20b, gpt120b, all (default)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
MODEL=${2:-all}  # Default to all models if not specified
CACHE_DIR="$HOME/.cache/llama.cpp"
BUILD_DIR="$(pwd)/build/bin"

# Model paths
GEMMA_3B="$CACHE_DIR/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
GPT_OSS_20B="$CACHE_DIR/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
GPT_OSS_120B="$CACHE_DIR/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"

# Benchmark parameters
PROMPT_SIZE=512
GEN_SIZE=128
REPETITIONS=3

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}     Single GPU Benchmark Suite (GPU $GPU_ID)     ${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Function to run benchmark
run_benchmark() {
    local model_path=$1
    local model_name=$2
    local ngl=$3

    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}[SKIP] $model_name not found at: $model_path${NC}"
        return
    fi

    echo -e "${GREEN}[BENCHMARK] $model_name${NC}"
    echo -e "Model path: $model_path"
    echo -e "GPU layers: $ngl"
    echo ""

    export HIP_VISIBLE_DEVICES=$GPU_ID
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    "$BUILD_DIR/llama-bench" \
        -m "$model_path" \
        -p $PROMPT_SIZE \
        -n $GEN_SIZE \
        -r $REPETITIONS \
        -ngl $ngl \
        2>/dev/null | grep -E "^\\||build:"

    echo ""
    echo -e "${GREEN}----------------------------------------${NC}"
    echo ""
}

# System info
echo -e "${YELLOW}System Information:${NC}"
if command -v rocm-smi &> /dev/null; then
    echo "GPU Backend: AMD ROCm/HIP"
    rocm-smi --showproductname 2>/dev/null | grep "GPU\[$GPU_ID\]" || true
else
    echo "GPU Backend: NVIDIA CUDA"
    nvidia-smi -i $GPU_ID --query-gpu=name --format=csv,noheader || true
fi
echo ""

# Run benchmarks based on model selection
echo -e "${BLUE}Starting benchmarks for: $MODEL${NC}"
echo ""

case "$MODEL" in
    gemma)
        run_benchmark "$GEMMA_3B" "Gemma 3-1B (Q4_K_M)" 99
        ;;
    gpt20b)
        run_benchmark "$GPT_OSS_20B" "GPT-OSS 20B (MXFP4)" 99
        ;;
    gpt120b)
        echo -e "${YELLOW}Note: GPT-OSS 120B may not fit on single GPU${NC}"
        run_benchmark "$GPT_OSS_120B" "GPT-OSS 120B (MXFP4)" 99
        ;;
    all)
        run_benchmark "$GEMMA_3B" "Gemma 3-1B (Q4_K_M)" 99
        run_benchmark "$GPT_OSS_20B" "GPT-OSS 20B (MXFP4)" 99
        echo -e "${YELLOW}Note: GPT-OSS 120B may not fit on single GPU${NC}"
        run_benchmark "$GPT_OSS_120B" "GPT-OSS 120B (MXFP4)" 99
        ;;
    *)
        echo -e "${RED}Error: Invalid model '$MODEL'${NC}"
        echo "Valid options: gemma, gpt20b, gpt120b, all"
        exit 1
        ;;
esac

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${BLUE}==================================================${NC}"