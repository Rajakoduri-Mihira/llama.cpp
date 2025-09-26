#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_TTNN_API __declspec(dllexport) extern
#        else
#            define GGML_TTNN_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_TTNN_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_TTNN_API extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Backend registry
GGML_TTNN_API ggml_backend_reg_t ggml_backend_ttnn_reg(void);

// Device management
GGML_TTNN_API int  ggml_backend_ttnn_get_device_count(void);
GGML_TTNN_API void ggml_backend_ttnn_get_device_description(int device, char * description, size_t description_size);

// Backend initialization
GGML_TTNN_API ggml_backend_buffer_type_t ggml_backend_ttnn_buffer_type(int device);
GGML_TTNN_API ggml_backend_buffer_type_t ggml_backend_ttnn_host_buffer_type(void);

// Split buffer for multi-device
GGML_TTNN_API ggml_backend_buffer_type_t ggml_backend_ttnn_split_buffer_type(int main_device, const float * tensor_split);

// Backend operations
GGML_TTNN_API bool ggml_backend_ttnn_register_host_buffer(void * buffer, size_t size);
GGML_TTNN_API void ggml_backend_ttnn_unregister_host_buffer(void * buffer);

#ifdef __cplusplus
}
#endif