#!/bin/bash
# Complete Benchmark Suite for llama.cpp
# Compares single vs multi-GPU performance across all models
# Usage: ./benchmark-suite.sh [OUTPUT_FILE]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_FILE=${1:-benchmark_results_$(date +%Y%m%d_%H%M%S).txt}
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

# Detect available GPUs
if command -v rocm-smi &> /dev/null; then
    NUM_GPUS=$(rocm-smi --showbus 2>/dev/null | grep -c "GPU\[" || echo "1")
    GPU_TYPE="AMD ROCm/HIP"
    GPU_CMD="rocm-smi"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
    GPU_TYPE="NVIDIA CUDA"
    GPU_CMD="nvidia-smi"
fi

ALL_GPUS=$(seq -s, 0 $((NUM_GPUS-1)))

# Function to run benchmark and capture results
run_benchmark_capture() {
    local model_path=$1
    local model_name=$2
    local gpu_config=$3
    local config_name=$4

    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}[SKIP] $model_name not found${NC}"
        return
    fi

    echo -e "${GREEN}Testing: $model_name ($config_name)${NC}" >&2

    export HIP_VISIBLE_DEVICES=$gpu_config
    export CUDA_VISIBLE_DEVICES=$gpu_config

    local benchmark_output=$("$BUILD_DIR/llama-bench" \
        -m "$model_path" \
        -p $PROMPT_SIZE \
        -n $GEN_SIZE \
        -r $REPETITIONS \
        -ngl 99 \
        2>/dev/null)

    local pp_result=$(echo "$benchmark_output" | grep "pp512" | grep -o '[0-9]\+\.[0-9]\+' | tail -2 | head -1)
    local tg_result=$(echo "$benchmark_output" | grep "tg128" | grep -o '[0-9]\+\.[0-9]\+' | tail -2 | head -1)

    echo "$pp_result $tg_result"
}

# Start benchmark suite
{
    echo "========================================="
    echo "    llama.cpp Benchmark Suite Results    "
    echo "========================================="
    echo ""
    echo "Date: $(date)"
    echo "GPU Type: $GPU_TYPE"
    echo "Number of GPUs: $NUM_GPUS"
    echo "Prompt size: $PROMPT_SIZE tokens"
    echo "Generation size: $GEN_SIZE tokens"
    echo "Repetitions: $REPETITIONS"
    echo ""

    if [ "$GPU_TYPE" = "AMD ROCm/HIP" ]; then
        echo "GPU Information:"
        rocm-smi --showproductname 2>/dev/null || true
    else
        echo "GPU Information:"
        nvidia-smi --list-gpus 2>/dev/null || true
    fi
    echo ""

    echo "========================================="
    echo "          Benchmark Results              "
    echo "========================================="
    echo ""

} | tee "$OUTPUT_FILE"

# Header for results table
{
    echo "Model Configuration Performance Comparison"
    echo "-------------------------------------------"
    printf "%-25s %-20s %-15s %-15s\n" "Model" "Configuration" "Prompt (t/s)" "Generation (t/s)"
    echo "-------------------------------------------"
} | tee -a "$OUTPUT_FILE"

# Test each model with different GPU configurations
echo -e "${CYAN}Running comprehensive benchmarks...${NC}"
echo ""

# Gemma 3B - Single GPU
if [ -f "$GEMMA_3B" ]; then
    echo -e "${BLUE}Testing Gemma 3-1B...${NC}"

    # Single GPU
    result=$(run_benchmark_capture "$GEMMA_3B" "Gemma 3-1B" "0" "Single GPU")
    pp=$(echo "$result" | awk '{print $1}')
    tg=$(echo "$result" | awk '{print $2}')
    printf "%-25s %-20s %-15s %-15s\n" "Gemma 3-1B (Q4_K_M)" "Single GPU (0)" "$pp" "$tg" | tee -a "$OUTPUT_FILE"

    # Multi-GPU (2)
    if [ $NUM_GPUS -ge 2 ]; then
        result=$(run_benchmark_capture "$GEMMA_3B" "Gemma 3-1B" "0,1" "2 GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "Gemma 3-1B (Q4_K_M)" "2 GPUs (0,1)" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi

    # All GPUs
    if [ $NUM_GPUS -ge 2 ]; then
        result=$(run_benchmark_capture "$GEMMA_3B" "Gemma 3-1B" "$ALL_GPUS" "All GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "Gemma 3-1B (Q4_K_M)" "All $NUM_GPUS GPUs" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
fi

# GPT-OSS 20B
if [ -f "$GPT_OSS_20B" ]; then
    echo -e "${BLUE}Testing GPT-OSS 20B...${NC}"

    # Single GPU
    result=$(run_benchmark_capture "$GPT_OSS_20B" "GPT-OSS 20B" "0" "Single GPU")
    pp=$(echo "$result" | awk '{print $1}')
    tg=$(echo "$result" | awk '{print $2}')
    printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 20B (MXFP4)" "Single GPU (0)" "$pp" "$tg" | tee -a "$OUTPUT_FILE"

    # Multi-GPU (2)
    if [ $NUM_GPUS -ge 2 ]; then
        result=$(run_benchmark_capture "$GPT_OSS_20B" "GPT-OSS 20B" "0,1" "2 GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 20B (MXFP4)" "2 GPUs (0,1)" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi

    # All GPUs
    if [ $NUM_GPUS -ge 2 ]; then
        result=$(run_benchmark_capture "$GPT_OSS_20B" "GPT-OSS 20B" "$ALL_GPUS" "All GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 20B (MXFP4)" "All $NUM_GPUS GPUs" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
fi

# GPT-OSS 120B (likely only works with multi-GPU)
if [ -f "$GPT_OSS_120B" ]; then
    echo -e "${BLUE}Testing GPT-OSS 120B...${NC}"

    # Try single GPU (will likely fail)
    echo -e "${YELLOW}Note: 120B model may not fit on single GPU${NC}"
    result=$(run_benchmark_capture "$GPT_OSS_120B" "GPT-OSS 120B" "0" "Single GPU" 2>/dev/null || echo "OOM")
    if [ "$result" != "OOM" ]; then
        pp=$(echo "$result" | grep pp512 | awk '{print $(NF-2)}' || echo "N/A")
        tg=$(echo "$result" | grep tg128 | awk '{print $(NF-2)}' || echo "N/A")
    else
        pp="OOM"
        tg="OOM"
    fi
    printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 120B (MXFP4)" "Single GPU (0)" "$pp" "$tg" | tee -a "$OUTPUT_FILE"

    # Multi-GPU configurations
    if [ $NUM_GPUS -ge 4 ]; then
        result=$(run_benchmark_capture "$GPT_OSS_120B" "GPT-OSS 120B" "0,1,2,3" "4 GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 120B (MXFP4)" "4 GPUs" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi

    # All GPUs
    if [ $NUM_GPUS -ge 2 ]; then
        result=$(run_benchmark_capture "$GPT_OSS_120B" "GPT-OSS 120B" "$ALL_GPUS" "All GPUs")
        pp=$(echo "$result" | awk '{print $1}')
        tg=$(echo "$result" | awk '{print $2}')
        printf "%-25s %-20s %-15s %-15s\n" "GPT-OSS 120B (MXFP4)" "All $NUM_GPUS GPUs" "$pp" "$tg" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
fi

{
    echo "-------------------------------------------"
    echo ""
    echo "Benchmark suite completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Legend:"
    echo "  pp512 = Prompt processing speed (tokens/second)"
    echo "  tg128 = Text generation speed (tokens/second)"
    echo "  OOM = Out of Memory"
    echo ""
} | tee -a "$OUTPUT_FILE"

echo -e "${GREEN}✓ All benchmarks complete!${NC}"
echo -e "${BLUE}Results saved to: ${YELLOW}$OUTPUT_FILE${NC}"