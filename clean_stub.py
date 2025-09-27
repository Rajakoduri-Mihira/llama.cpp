#!/usr/bin/env python3
"""Remove TTNN_STUB_IMPLEMENTATION conditionals and keep only the real implementation."""

import re

def clean_stub_code(content):
    """Remove stub implementation code blocks and keep real implementation."""

    # Pattern to match #ifndef TTNN_STUB_IMPLEMENTATION ... #else ... #endif
    pattern = r'#ifndef TTNN_STUB_IMPLEMENTATION\n(.*?)(?:#else.*?)?#endif // TTNN_STUB_IMPLEMENTATION'

    def replacer(match):
        # Keep the content inside #ifndef (real implementation)
        return match.group(1).rstrip()

    # Apply the pattern with DOTALL flag to match across lines
    cleaned = re.sub(pattern, replacer, content, flags=re.DOTALL)

    # Remove #else blocks with stub implementation
    pattern2 = r'#else // TTNN_STUB_IMPLEMENTATION.*?#endif // TTNN_STUB_IMPLEMENTATION'
    cleaned = re.sub(pattern2, '', cleaned, flags=re.DOTALL)

    # Remove standalone stub struct definitions
    pattern3 = r'#else // TTNN_STUB_IMPLEMENTATION\n\n// Stub implementations.*?(?=#endif)'
    cleaned = re.sub(pattern3, '', cleaned, flags=re.DOTALL)

    return cleaned

# Read the file
with open('/workspaces/oxpython/llama.cpp/ggml/src/ggml-ttnn/ggml-ttnn.cpp', 'r') as f:
    content = f.read()

# Clean the content
cleaned = clean_stub_code(content)

# Write back
with open('/workspaces/oxpython/llama.cpp/ggml/src/ggml-ttnn/ggml-ttnn.cpp', 'w') as f:
    f.write(cleaned)

print("Cleaned stub implementations from ggml-ttnn.cpp")