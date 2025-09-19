#!/bin/bash
# Simple working chat script for llama.cpp models
# Usage: ./simple-chat.sh "your prompt here" [model]

PROMPT="${1:-Write a simple Python hello world program}"
MODEL="${2:-gpt20b}"  # Default to working GPT-20B

case "$MODEL" in
    gemma)
        MODEL_PATH="$HOME/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
        GPU_CONFIG="0"  # Single GPU optimal
        ;;
    gpt20b)
        MODEL_PATH="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
        GPU_CONFIG="0"  # Single GPU optimal
        ;;
    gpt120b)
        MODEL_PATH="$HOME/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"
        GPU_CONFIG="0,1,2,3,4,5"  # Requires all GPUs
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: gemma, gpt20b, gpt120b"
        exit 1
        ;;
esac

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    echo "Download it first with llama-cli -hf"
    exit 1
fi

echo "🤖 Using $MODEL model..."
echo "📝 Prompt: $PROMPT"
echo "⚡ Processing..."
echo ""

export HIP_VISIBLE_DEVICES=$GPU_CONFIG
export CUDA_VISIBLE_DEVICES=$GPU_CONFIG

# Run with minimal output
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -n 256 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    --log-disable \
    -p "$PROMPT" \

echo ""
echo "✅ Done!"