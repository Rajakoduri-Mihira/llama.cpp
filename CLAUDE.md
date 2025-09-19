# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Primary build system:** CMake (Makefile is deprecated)

```bash
# Standard CPU-only build
cmake -B build
cmake --build build --config Release -j $(nproc)

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Metal build (macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)
```

Built binaries are located in `build/bin/`

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure -j $(nproc)

# Server unit tests (requires Python environment)
cd tools/server/tests
source ../../../.venv/bin/activate  # If .venv exists
./tests.sh
```

## Code Formatting and Linting

**C++ code:** Always format before committing
```bash
git clang-format
```

**Python code:** Use flake8 (max-line-length=125)
```bash
source .venv/bin/activate  # If .venv exists
flake8
```

**Pre-commit hooks:**
```bash
pre-commit run --all-files
```

## Architecture Overview

### Core Components

- **`src/llama.cpp`**: Main library implementation (~8000 lines)
- **`include/llama.h`**: Primary C API header (~2000 lines)
- **`ggml/`**: Vendored tensor library (core dependency)
- **`common/`**: Shared utilities for examples
- **`examples/`**: 30+ example applications

### Key Source Files

- `src/llama-*.cpp`: Modular components (adapter, arch, batch, chat, context, grammar, etc.)
- `include/llama.h`: Public C API
- `CMakeLists.txt`: Build configuration

### Primary Executables

Located in `build/bin/` after building:
- `llama-cli`: Main inference tool
- `llama-server`: OpenAI-compatible HTTP server
- `llama-quantize`: Model quantization
- `llama-perplexity`: Model evaluation
- `llama-bench`: Performance benchmarking

## Development Workflow

1. **Before making changes:** Understand existing code style and conventions
2. **After code changes:**
   - Format: `git clang-format`
   - Build: `cmake --build build --config Release`
   - Test: `ctest --test-dir build --output-on-failure`
3. **For server changes:** Also run server unit tests
4. **Performance validation:** Use `llama-bench` and `llama-perplexity`

## Important Notes

- **Backends:** CPU (always available), CUDA, Metal, Vulkan, SYCL, ROCm, MUSA
- **Dependencies:** Minimal by design - mainly OpenMP and optionally libcurl
- **Build time:** ~10 minutes with ccache, ~25 minutes without
- **Test suite:** 38 tests, ~30 seconds runtime (2-3 may fail without network)
- **CI trigger:** Add "ggml-ci" to commit message for heavy CI workloads

## Quick Validation Commands

```bash
# Test basic functionality
./build/bin/llama-cli --version

# Test with model (requires .gguf file)
./build/bin/llama-cli -m model.gguf -p "Hello" -n 10

# Benchmark
./build/bin/llama-bench -m model.gguf
```