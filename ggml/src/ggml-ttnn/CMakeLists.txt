cmake_minimum_required(VERSION 3.24)

# TT-Metal setup using tenstorrent-sdk source
set(TT_METAL_SDK_PATH "/workspaces/oxpython/tenstorrent-sdk")
set(TT_METAL_HOME $ENV{TT_METAL_HOME})

if(NOT TT_METAL_HOME)
    if(EXISTS ${TT_METAL_SDK_PATH})
        set(TT_METAL_HOME ${TT_METAL_SDK_PATH})
    else()
        set(TT_METAL_DEFAULT_PATH "/opt/oxpython/external/tt-metal")
        if(EXISTS ${TT_METAL_DEFAULT_PATH})
            set(TT_METAL_HOME ${TT_METAL_DEFAULT_PATH})
        else()
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

set(ENV{TT_METAL_HOME} "${TT_METAL_HOME}")
set(ENV{PYTHONPATH} "${TT_METAL_HOME}:$ENV{PYTHONPATH}")

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development)

# TT-Metal include directories - prefer OxPython's pre-built includes
set(OXPYTHON_TT_METAL_PATH "/workspaces/oxpython/components/backends/tenstorrent/externals/tt-metal")

file(GLOB TT_METAL_SPDLOG_DIRS
    "${OXPYTHON_TT_METAL_PATH}/.cpmcache/spdlog/*/include"
    "${TT_METAL_HOME}/.cpmcache/spdlog/*/include"
)

file(GLOB TT_METAL_FMT_DIRS
    "${OXPYTHON_TT_METAL_PATH}/.cpmcache/fmt/*/include"
    "${TT_METAL_HOME}/.cpmcache/fmt/*/include"
)

set(TT_METAL_INCLUDE_DIRS
    # OxPython pre-built includes (preferred)
    "${OXPYTHON_TT_METAL_PATH}/build_Release/include"
    "${OXPYTHON_TT_METAL_PATH}"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/cpp"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/cpp/ttnn"
    "${OXPYTHON_TT_METAL_PATH}/tt_metal"
    "${OXPYTHON_TT_METAL_PATH}/tt_metal/include"
    "${OXPYTHON_TT_METAL_PATH}/build_Release/include/metalium-thirdparty"
    "${OXPYTHON_TT_METAL_PATH}/build_Release/_deps/spdlog-src/include"
    "${OXPYTHON_TT_METAL_PATH}/build_Release/_deps/fmt-src/include"
    # Fallback to source SDK
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/ttnn/cpp"
    "${TT_METAL_HOME}/ttnn/cpp/ttnn"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/include"
    "${TT_METAL_HOME}/tt_metal/impl"
    "${TT_METAL_HOME}/tt_metal/hw/inc"
    "${TT_METAL_HOME}/third_party"
    "${TT_METAL_HOME}/build_Release/include/metalium-thirdparty"
    # Also check for built includes if available
    "${TT_METAL_HOME}/build_Release/include"
    "${TT_METAL_HOME}/build/include"
    "${TT_METAL_HOME}/build_Release/_deps/spdlog-src/include"
    "${TT_METAL_HOME}/build_Release/_deps/fmt-src/include"
    ${TT_METAL_SPDLOG_DIRS}
    ${TT_METAL_FMT_DIRS}
)

set(OXPYTHON_TT_METAL_PATH "/workspaces/oxpython/components/backends/tenstorrent/externals/tt-metal")

set(TTNN_HEADER_CANDIDATES
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/ttnn"
    "${TT_METAL_HOME}/ttnn/api"
    "${OXPYTHON_TT_METAL_PATH}"
    "${OXPYTHON_TT_METAL_PATH}/ttnn"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/api"
)

find_path(TTNN_PUBLIC_INCLUDE_DIR
    NAMES ttnn/tensor/tensor.hpp ttnn/api/ttnn/tensor/tensor.hpp
    PATHS ${TTNN_HEADER_CANDIDATES}
    NO_DEFAULT_PATH
)

set(TTNN_RUNTIME_CANDIDATES
    "${OXPYTHON_TT_METAL_PATH}/build_Release/lib"
    "${TT_METAL_HOME}/build_Release/lib"
    "${TT_METAL_HOME}/build/lib"
)

find_library(TTNN_METAL_LIB
    NAMES tt_metal
    PATHS ${TTNN_RUNTIME_CANDIDATES}
    NO_DEFAULT_PATH
)

find_library(TTNN_STL_LIB
    NAMES tt_stl
    PATHS ${TTNN_RUNTIME_CANDIDATES}
    NO_DEFAULT_PATH
)

find_library(TTNN_CORE_LIB
    NAMES ttnn_core
    PATHS ${TTNN_RUNTIME_CANDIDATES}
    NO_DEFAULT_PATH
)

set(TTNN_CPP_LIB "TTNN_CPP_LIB-NOTFOUND")
set(TTNN_CPP_CANDIDATES
    "${OXPYTHON_TT_METAL_PATH}/build_Release/lib/_ttnncpp.so"
    "${TT_METAL_HOME}/build_Release/lib/_ttnncpp.so"
    "${OXPYTHON_TT_METAL_PATH}/ttnn/_ttnncpp.so"
    "${TT_METAL_HOME}/ttnn/_ttnncpp.so"
)

foreach(cand ${TTNN_CPP_CANDIDATES})
    if(EXISTS "${cand}")
        set(TTNN_CPP_LIB "${cand}")
        break()
    endif()
endforeach()

set(GGML_TTNN_SOURCES ggml-ttnn.cpp)

# Check if we have the required TTNN runtime components
if(NOT TTNN_PUBLIC_INCLUDE_DIR OR NOT TTNN_METAL_LIB)
    message(FATAL_ERROR "TTNN runtime prerequisites not found. Please ensure TT_METAL_HOME is set correctly.")
endif()

ggml_add_backend_library(ggml-ttnn ${GGML_TTNN_SOURCES})

target_include_directories(ggml-ttnn PRIVATE
    ${TT_METAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${Python3_INCLUDE_DIRS}
    ${TTNN_PUBLIC_INCLUDE_DIR}
)

if(TTNN_CORE_LIB)
    target_link_libraries(ggml-ttnn PRIVATE ${TTNN_CORE_LIB})
endif()

if(TTNN_CPP_LIB)
    target_link_libraries(ggml-ttnn PRIVATE ${TTNN_CPP_LIB})
endif()

target_link_libraries(ggml-ttnn PRIVATE
    ${TTNN_METAL_LIB}
    ${TTNN_STL_LIB}
)

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
