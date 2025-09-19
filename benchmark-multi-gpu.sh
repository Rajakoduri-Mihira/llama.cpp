#!/bin/bash
# Multi-GPU Benchmark Script for llama.cpp models
# Usage: ./benchmark-multi-gpu.sh [GPU_LIST] [MODEL]
# GPU_LIST: comma-separated GPU IDs (e.g., "0,1,2,3") or "all" for all GPUs
# MODEL: gemma, gpt20b, gpt120b, all (default)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GPU_LIST=${1:-all}  # Default to all GPUs if not specified
MODEL=${2:-all}     # Default to all models if not specified
CACHE_DIR="$HOME/.cache/llama.cpp"
BUILD_DIR="$(pwd)/build/bin"

# Detect available GPUs
if command -v rocm-smi &> /dev/null; then
    NUM_GPUS=$(rocm-smi --showbus 2>/dev/null | grep -c "GPU\[" || echo "1")
    GPU_TYPE="AMD ROCm/HIP"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
    GPU_TYPE="NVIDIA CUDA"
fi

# Process GPU list
if [ "$GPU_LIST" = "all" ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
else
    GPU_IDS="$GPU_LIST"
fi

# Model paths
GEMMA_3B="$CACHE_DIR/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
GPT_OSS_20B="$CACHE_DIR/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
GPT_OSS_120B="$CACHE_DIR/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"

# Benchmark parameters
PROMPT_SIZE=512
GEN_SIZE=128
REPETITIONS=3

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    Multi-GPU Benchmark Suite (GPUs: $GPU_IDS)    ${NC}"
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
    echo -e "GPUs used: $GPU_IDS"
    echo ""

    export HIP_VISIBLE_DEVICES=$GPU_IDS
    export CUDA_VISIBLE_DEVICES=$GPU_IDS

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

# Function to count GPUs in list
count_gpus() {
    echo "$1" | tr ',' ' ' | wc -w
}

# System info
echo -e "${YELLOW}System Information:${NC}"
echo "GPU Backend: $GPU_TYPE"
echo "Total GPUs available: $NUM_GPUS"
echo "GPUs selected: $GPU_IDS ($(count_gpus $GPU_IDS) GPUs)"
echo ""

if command -v rocm-smi &> /dev/null; then
    echo "GPU Details:"
    for gpu in $(echo $GPU_IDS | tr ',' ' '); do
        rocm-smi --showproductname 2>/dev/null | grep "GPU\[$gpu\]" || true
    done
else
    echo "GPU Details:"
    for gpu in $(echo $GPU_IDS | tr ',' ' '); do
        nvidia-smi -i $gpu --query-gpu=name --format=csv,noheader || true
    done
fi
echo ""

# Run benchmarks based on model selection
echo -e "${BLUE}Starting multi-GPU benchmarks for: $MODEL${NC}"
echo ""

case "$MODEL" in
    gemma)
        run_benchmark "$GEMMA_3B" "Gemma 3-1B (Q4_K_M)" 99
        ;;
    gpt20b)
        run_benchmark "$GPT_OSS_20B" "GPT-OSS 20B (MXFP4)" 99
        ;;
    gpt120b)
        run_benchmark "$GPT_OSS_120B" "GPT-OSS 120B (MXFP4)" 99
        ;;
    all)
        run_benchmark "$GEMMA_3B" "Gemma 3-1B (Q4_K_M)" 99
        run_benchmark "$GPT_OSS_20B" "GPT-OSS 20B (MXFP4)" 99
        run_benchmark "$GPT_OSS_120B" "GPT-OSS 120B (MXFP4)" 99
        ;;
    *)
        echo -e "${RED}Error: Invalid model '$MODEL'${NC}"
        echo "Valid options: gemma, gpt20b, gpt120b, all"
        exit 1
        ;;
esac

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Multi-GPU Benchmark Complete!${NC}"
echo -e "${BLUE}==================================================${NC}"