#!/bin/bash
# Interactive Model Launcher for llama.cpp
# Usage: ./launch-model.sh [MODEL] [GPU_LIST] [MODE]
# MODEL: gemma, gpt20b, gpt120b
# GPU_LIST: single GPU ID, comma-separated list, or "all"
# MODE: cli (default), server, interactive

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MODEL=${1:-gemma}
GPU_LIST=${2:-0}
MODE=${3:-cli}
CACHE_DIR="$HOME/.cache/llama.cpp"
BUILD_DIR="$(pwd)/build/bin"

# Model configurations
declare -A MODEL_PATHS
MODEL_PATHS["gemma"]="$CACHE_DIR/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
MODEL_PATHS["gpt20b"]="$CACHE_DIR/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
MODEL_PATHS["gpt120b"]="$CACHE_DIR/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"

declare -A MODEL_NAMES
MODEL_NAMES["gemma"]="Gemma 3-1B (Q4_K_M)"
MODEL_NAMES["gpt20b"]="GPT-OSS 20B (MXFP4)"
MODEL_NAMES["gpt120b"]="GPT-OSS 120B (MXFP4)"

# Recommended settings per model
declare -A MODEL_CTX
MODEL_CTX["gemma"]=8192
MODEL_CTX["gpt20b"]=8192
MODEL_CTX["gpt120b"]=4096

declare -A MODEL_NGL
MODEL_NGL["gemma"]=99
MODEL_NGL["gpt20b"]=99
MODEL_NGL["gpt120b"]=99

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

# Validate model
if [ -z "${MODEL_PATHS[$MODEL]}" ]; then
    echo -e "${RED}Error: Invalid model '$MODEL'${NC}"
    echo "Valid options: gemma, gpt20b, gpt120b"
    exit 1
fi

MODEL_PATH="${MODEL_PATHS[$MODEL]}"
MODEL_NAME="${MODEL_NAMES[$MODEL]}"
CTX_SIZE="${MODEL_CTX[$MODEL]}"
NGL="${MODEL_NGL[$MODEL]}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at: $MODEL_PATH${NC}"
    echo "Please download the model first using llama-cli"
    exit 1
fi

# Display launch info
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}          llama.cpp Model Launcher                ${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo -e "${GREEN}Model:${NC} $MODEL_NAME"
echo -e "${GREEN}Path:${NC} $MODEL_PATH"
echo -e "${GREEN}GPUs:${NC} $GPU_IDS"
echo -e "${GREEN}Mode:${NC} $MODE"
echo -e "${GREEN}Context:${NC} $CTX_SIZE tokens"
echo ""

# Set GPU environment
export HIP_VISIBLE_DEVICES=$GPU_IDS
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Suppress debug output
export GGML_CUDA_NO_PEER_COPY=1

# Launch based on mode
case "$MODE" in
    cli)
        echo -e "${YELLOW}Launching CLI mode...${NC}"

        # GPT-OSS models need different settings
        if [[ "$MODEL" == "gpt20b" ]] || [[ "$MODEL" == "gpt120b" ]]; then
            echo -e "${YELLOW}Enter your prompt (press Enter to submit):${NC}"
            echo ""
            read -p "> " USER_PROMPT
            echo -e "${CYAN}Loading model (this may take 20+ seconds)...${NC}"

            # Suppress stderr loading messages, keep stdout
            TEMP_ERR=$(mktemp)
            trap "rm -f $TEMP_ERR" EXIT

            echo ""

            # Suppress GGML debug output
            export GGML_DEBUG=0
            export LLAMA_LOG_LEVEL=error

            # Run and capture all output
            OUTPUT_FILE=$(mktemp)
            trap "rm -f $OUTPUT_FILE $TEMP_ERR" EXIT

            # Format prompt for better Q&A response
            FULL_PROMPT="Question: $USER_PROMPT
Answer:"

            "$BUILD_DIR/llama-cli" \
                -m "$MODEL_PATH" \
                -n 512 \
                -c $CTX_SIZE \
                -ngl $NGL \
                --temp 0.7 \
                --repeat-penalty 1.1 \
                --no-display-prompt \
                --log-verbosity 0 \
                -no-cnv \
                --simple-io \
                -p "$FULL_PROMPT" > $OUTPUT_FILE 2>&1

            # Extract just the generated text
            echo -e "${GREEN}Response:${NC}"

            # Extract generated text - find content after all the setup messages
            # The actual generation appears after the sampler chain info
            sed -n '/^generate: n_ctx/,/^llama_perf/p' $OUTPUT_FILE | \
            sed '1d;$d' | \
            grep -v "^$" | \
            head -20

            # Show performance stats
            echo ""
            echo -e "${CYAN}=== Performance Stats ===${NC}"
            grep "llama_perf" $OUTPUT_FILE | sed 's/^llama_perf_/  /'

            # Extract clean tokens/sec
            EVAL_SPEED=$(grep "eval time" $OUTPUT_FILE | tail -1 | sed -n 's/.*(\([0-9.]*\) tokens per second.*/\1/p')
            if [ -n "$EVAL_SPEED" ]; then
                echo ""
                echo -e "${YELLOW}  Generation speed: $EVAL_SPEED tokens/sec${NC}"
            fi
            echo ""
        else
            echo -e "${YELLOW}Type your prompt and press Ctrl+D when done${NC}"
            echo ""
            "$BUILD_DIR/llama-cli" \
                -m "$MODEL_PATH" \
                -n 512 \
                -c $CTX_SIZE \
                -ngl $NGL \
                --color \
                --log-disable \
                -p "You are a helpful AI assistant."
        fi
        ;;

    interactive)
        echo -e "${YELLOW}Launching interactive mode...${NC}"

        # GPT-OSS models need different interactive settings
        if [[ "$MODEL" == "gpt20b" ]] || [[ "$MODEL" == "gpt120b" ]]; then
            echo -e "${YELLOW}Interactive mode for GPT-OSS models${NC}"
            echo -e "${YELLOW}Type your question and press Enter twice to submit${NC}"
            echo -e "${YELLOW}Type 'exit' or Ctrl+C to quit${NC}"
            echo ""
            "$BUILD_DIR/llama-cli" \
                -m "$MODEL_PATH" \
                -n 512 \
                -c $CTX_SIZE \
                -ngl $NGL \
                --temp 0.7 \
                --repeat-penalty 1.1 \
                --color \
                --interactive \
                --interactive-first \
                --simple-io \
                --log-disable \
                -r "Q:" \
                --in-prefix " " \
                --in-suffix "A:" \
                -p "Answer questions concisely.

Q: "
        else
            echo -e "${YELLOW}Type 'exit' or Ctrl+C to quit${NC}"
            echo ""
            "$BUILD_DIR/llama-cli" \
                -m "$MODEL_PATH" \
                -n 512 \
                -c $CTX_SIZE \
                -ngl $NGL \
                --color \
                --interactive \
                --interactive-first \
                --log-disable \
                -r "User:" \
                -p "You are a helpful AI assistant. Respond concisely to the user's questions.

User:"
        fi
        ;;

    server)
        echo -e "${YELLOW}Launching server mode...${NC}"
        echo -e "${GREEN}Server will be available at: http://localhost:8080${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        echo ""

        # Determine port based on model to allow multiple servers
        case "$MODEL" in
            gemma)  PORT=8080 ;;
            gpt20b) PORT=8081 ;;
            gpt120b) PORT=8082 ;;
        esac

        "$BUILD_DIR/llama-server" \
            -m "$MODEL_PATH" \
            -c $CTX_SIZE \
            -ngl $NGL \
            --host 0.0.0.0 \
            --port $PORT \
            --n-predict 512 \
            --threads $(nproc)
        ;;

    *)
        echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
        echo "Valid options: cli, interactive, server"
        exit 1
        ;;
esac