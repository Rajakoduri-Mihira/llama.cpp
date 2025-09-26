#!/bin/bash
# Setup script for TTNN backend in llama.cpp

echo "Setting up Tenstorrent TTNN backend environment..."

# Detect TT-Metal location
if [ -z "$TT_METAL_HOME" ]; then
    # Try default location
    if [ -d "/opt/oxpython/external/tt-metal" ]; then
        export TT_METAL_HOME="/opt/oxpython/external/tt-metal"
    elif [ -d "../tenstorrent-sdk" ]; then
        export TT_METAL_HOME="$(realpath ../tenstorrent-sdk)"
    elif [ -d "../components/backends/tenstorrent/externals/tt-metal" ]; then
        export TT_METAL_HOME="$(realpath ../components/backends/tenstorrent/externals/tt-metal)"
    else
        echo "Warning: TT-Metal SDK not found. Please set TT_METAL_HOME manually."
        echo "Expected locations:"
        echo "  - /opt/oxpython/external/tt-metal"
        echo "  - ../tenstorrent-sdk"
        echo ""
        echo "For stub testing without actual hardware, you can continue."
    fi
fi

if [ -n "$TT_METAL_HOME" ]; then
    echo "TT_METAL_HOME set to: $TT_METAL_HOME"
    export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"

    # Check if we need to build TT-Metal
    if [ -d "$TT_METAL_HOME" ] && [ ! -f "$TT_METAL_HOME/build/lib/libtt_metal.so" ]; then
        echo "TT-Metal libraries not found. You may need to build them first."
        echo "Run: cd $TT_METAL_HOME && make"
    fi
fi

# Set compilation flags for TTNN backend
export GGML_TTNN=ON

echo ""
echo "Environment setup complete!"
echo ""
echo "To build llama.cpp with TTNN backend:"
echo "  cmake -B build -DGGML_TTNN=ON -DGGML_CURL=OFF"
echo "  cmake --build build --config Release -j \$(nproc)"
echo ""
echo "To test with TinyLlama:"
echo "  ./build/bin/llama-cli -m tinyllama.gguf -p \"Hello\" -n 50"
echo ""

# Print backend availability check
echo "To check if TTNN backend is available after building:"
echo "  ./build/bin/llama-cli --list-devices"