cmake_minimum_required(VERSION 3.24)

# TT-Metal setup using tenstorrent-sdk source
set(TT_METAL_SDK_PATH "/workspaces/oxpython/tenstorrent-sdk")
set(TT_METAL_HOME $ENV{TT_METAL_HOME})

if(NOT TT_METAL_HOME)
    # Try tenstorrent-sdk source first (preferred)
    if(EXISTS ${TT_METAL_SDK_PATH})
        set(TT_METAL_HOME ${TT_METAL_SDK_PATH})
    else()
        # Fall back to default paths
        set(TT_METAL_DEFAULT_PATH "/opt/oxpython/external/tt-metal")
        if(EXISTS ${TT_METAL_DEFAULT_PATH})
            set(TT_METAL_HOME ${TT_METAL_DEFAULT_PATH})
        else()
            # Try components backend path
            set(TT_METAL_RELATIVE_PATH "${CMAKE_SOURCE_DIR}/../../../components/backends/tenstorrent/externals/tt-metal")
            if(EXISTS ${TT_METAL_RELATIVE_PATH})
                set(TT_METAL_HOME ${TT_METAL_RELATIVE_PATH})
            endif()
        endif()
    endif()
endif()

if(NOT EXISTS "${TT_METAL_HOME}")
    message(FATAL_ERROR "TT-Metal SDK not found. Please set TT_METAL_HOME environment variable or ensure SDK is at ${TT_METAL_DEFAULT_PATH}")
endif()

message(STATUS "Using TT-Metal from: ${TT_METAL_HOME}")

# Set environment variables needed for TT-Metal
set(ENV{TT_METAL_HOME} "${TT_METAL_HOME}")
set(ENV{PYTHONPATH} "${TT_METAL_HOME}:$ENV{PYTHONPATH}")

# Find Python3 (required for TT-Metal)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development)

# TT-Metal include directories - prefer OxPython's pre-built includes
set(OXPYTHON_TT_METAL_PATH "/workspaces/oxpython/components/backends/tenstorrent/externals/tt-metal")
set(TT_METAL_INCLUDE_DIRS
    # OxPython pre-built includes (preferred)
    "${OXPYTHON_TT_METAL_PATH}/build_Release/include"
    "${OXPYTHON_TT_METAL_PATH}"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/cpp"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/cpp/ttnn"
    "${OXPYTHON_TT_METAL_PATH}/tt_metal"
    "${OXPYTHON_TT_METAL_PATH}/tt_metal/include"
    # Fallback to source SDK
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/ttnn/cpp"
    "${TT_METAL_HOME}/ttnn/cpp/ttnn"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/include"
    "${TT_METAL_HOME}/tt_metal/impl"
    "${TT_METAL_HOME}/tt_metal/hw/inc"
    "${TT_METAL_HOME}/third_party"
    # Also check for built includes if available
    "${TT_METAL_HOME}/build_Release/include"
    "${TT_METAL_HOME}/build/include"
)

# Try to find pre-built libraries
# First check OxPython's pre-built libraries
set(OXPYTHON_TT_METAL_PATH "/workspaces/oxpython/components/backends/tenstorrent/externals/tt-metal")

# TTNN library has unconventional name _ttnn.so, so check manually
set(TTNN_LIB_PATHS
    "${OXPYTHON_TT_METAL_PATH}/build_Release/lib/_ttnn.so"
    "${OXPYTHON_TT_METAL_PATH}/build_Release/lib/_ttnncpp.so"
    "${TT_METAL_HOME}/build/lib/_ttnn.so"
    "${TT_METAL_HOME}/build_Release/lib/_ttnn.so"
    "${TT_METAL_HOME}/ttnn/_ttnn.so"
)

set(TTNN_LIB "TTNN_LIB-NOTFOUND")
foreach(lib_path ${TTNN_LIB_PATHS})
    if(EXISTS "${lib_path}")
        set(TTNN_LIB "${lib_path}")
        message(STATUS "Found TTNN library: ${lib_path}")
        break()
    endif()
endforeach()

# Find TT Metal library
set(TT_METAL_LIB_PATHS
    "${OXPYTHON_TT_METAL_PATH}/build_Release/lib/libtt_metal.so"
    "${TT_METAL_HOME}/build/lib/libtt_metal.so"
    "${TT_METAL_HOME}/build_Release/lib/libtt_metal.so"
)

set(TT_METAL_LIB "TT_METAL_LIB-NOTFOUND")
foreach(lib_path ${TT_METAL_LIB_PATHS})
    if(EXISTS "${lib_path}")
        set(TT_METAL_LIB "${lib_path}")
        message(STATUS "Found TT-Metal library: ${lib_path}")
        break()
    endif()
endforeach()

# Create TTNN backend library
# Always use the full implementation, but conditionally compile based on libraries
set(GGML_TTNN_SOURCES ggml-ttnn.cpp)

if(NOT (TTNN_LIB AND TT_METAL_LIB))
    message(STATUS "TTNN libraries not found, using stub implementation mode")
else()
    message(STATUS "TTNN libraries found - backend will attempt dynamic loading")
endif()

ggml_add_backend_library(ggml-ttnn ${GGML_TTNN_SOURCES})

# Configure include directories
target_include_directories(ggml-ttnn PRIVATE
    ${TT_METAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${Python3_INCLUDE_DIRS}
)

# Link libraries
if(TTNN_LIB AND TT_METAL_LIB)
    message(STATUS "Found pre-built TT-Metal libraries - will load dynamically")
    # We found the libraries, but due to header complexity, we use dynamic loading
    # The libraries will be loaded at runtime via dlopen
else()
    message(STATUS "TT-Metal libraries not found, will use stub implementation")
endif()

# Always use stub compilation mode for now due to header dependencies
# The actual libraries are loaded dynamically at runtime
target_compile_definitions(ggml-ttnn PRIVATE TTNN_STUB_IMPLEMENTATION)

# Link with standard libraries
target_link_libraries(ggml-ttnn PRIVATE
    ${CMAKE_DL_LIBS}
    pthread
)

# Set C++ standard
target_compile_features(ggml-ttnn PRIVATE cxx_std_20)

# Add compile definitions
target_compile_definitions(ggml-ttnn PRIVATE
    GGML_TTNN_BACKEND
)

# If building shared library
if(BUILD_SHARED_LIBS)
    set_target_properties(ggml-ttnn PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Install targets
install(TARGETS ggml-ttnn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)