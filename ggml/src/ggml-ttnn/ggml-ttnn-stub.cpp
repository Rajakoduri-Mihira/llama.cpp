// Stub implementation for TTNN backend - allows compilation without TT-Metal SDK
// This file provides minimal implementations for Phase 1 testing

#include "ggml-ttnn.h"
#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include <cstring>
#include <memory>
#include <vector>
#include <cstdio>
#include <cassert>

#define GGML_TTNN_NAME "TTNN"
#define UNUSED GGML_UNUSED

namespace {

// Stub device structure
struct ttnn_device {
    int device_id;
    bool initialized;

    ttnn_device(int id) : device_id(id), initialized(false) {}

    bool initialize() {
        initialized = true;
        fprintf(stderr, "TTNN: Initialized stub device %d\n", device_id);
        return true;
    }

    void shutdown() {
        initialized = false;
    }
};

// Context structure
struct ttnn_context {
    int device_id;
    ttnn_device* device;

    ttnn_context(int id) : device_id(id), device(nullptr) {}
};

// Buffer type context
struct ttnn_buffer_type_context {
    int device_id;
    char name[32];
};

// Buffer context
struct ttnn_buffer_context {
    int device_id;
    void* data;
    size_t size;

    ttnn_buffer_context(int id, size_t sz) : device_id(id), size(sz) {
        data = malloc(size);
        if (!data) {
            throw std::bad_alloc();
        }
    }

    ~ttnn_buffer_context() {
        if (data) {
            free(data);
        }
    }
};

static std::vector<std::unique_ptr<ttnn_device>> g_devices;
static bool g_ttnn_initialized = false;

// Initialize stub backend
static bool ttnn_init() {
    if (g_ttnn_initialized) {
        return true;
    }

    // Create one stub device
    auto dev = std::make_unique<ttnn_device>(0);
    if (dev->initialize()) {
        g_devices.push_back(std::move(dev));
        g_ttnn_initialized = true;
    }

    return g_ttnn_initialized;
}

// Buffer type interface
static const char* ttnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ttnn_buffer_type_context* ctx = (ttnn_buffer_type_context*)buft->context;
    return ctx->name;
}

static ggml_backend_buffer_t ttnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ttnn_buffer_type_context* type_ctx = (ttnn_buffer_type_context*)buft->context;

    ttnn_buffer_context* ctx = new ttnn_buffer_context(type_ctx->device_id, size);

    ggml_backend_buffer_i iface = {};
    iface.free_buffer = [](ggml_backend_buffer_t buffer) {
        delete (ttnn_buffer_context*)buffer->context;
    };
    iface.get_base = [](ggml_backend_buffer_t buffer) -> void* {
        return ((ttnn_buffer_context*)buffer->context)->data;
    };
    iface.clear = [](ggml_backend_buffer_t buffer, uint8_t value) {
        ttnn_buffer_context* ctx = (ttnn_buffer_context*)buffer->context;
        memset(ctx->data, value, ctx->size);
    };

    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

static size_t ttnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 128;
}

// Backend interface
static const char* ttnn_backend_get_name(ggml_backend_t backend) {
    UNUSED(backend);
    return GGML_TTNN_NAME;
}

static void ttnn_backend_free(ggml_backend_t backend) {
    ttnn_context* ctx = (ttnn_context*)backend->context;
    delete ctx;
}

static enum ggml_status ttnn_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    UNUSED(backend);

    // Stub: just mark as success
    fprintf(stderr, "TTNN: Stub graph compute called with %d nodes\n", cgraph->n_nodes);
    return GGML_STATUS_SUCCESS;
}

// Device interface
static const char* ttnn_device_get_name(ggml_backend_dev_t dev) {
    int device_id = (intptr_t)dev->context;
    static char name[32];
    snprintf(name, sizeof(name), "TTNN%d", device_id);
    return name;
}

static const char* ttnn_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Tenstorrent TTNN stub device";
}

static void ttnn_device_get_memory(ggml_backend_dev_t dev, size_t* free, size_t* total) {
    UNUSED(dev);
    *free = 8ULL * 1024 * 1024 * 1024;
    *total = 8ULL * 1024 * 1024 * 1024;
}

static enum ggml_backend_dev_type ttnn_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ttnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props* props) {
    props->name = ttnn_device_get_name(dev);
    props->description = ttnn_device_get_description(dev);
    ttnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = ttnn_device_get_type(dev);
    props->caps = {};
}

static ggml_backend_t ttnn_device_init_backend(ggml_backend_dev_t dev, const char* params) {
    UNUSED(params);

    int device_id = (intptr_t)dev->context;
    ttnn_context* ctx = new ttnn_context(device_id);

    if (device_id < (int)g_devices.size()) {
        ctx->device = g_devices[device_id].get();
    }

    ggml_backend_i iface = {};
    iface.get_name = ttnn_backend_get_name;
    iface.free = ttnn_backend_free;
    iface.graph_compute = ttnn_backend_graph_compute;

    static ggml_guid guid = {0x54, 0x54, 0x4E, 0x4E, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ggml_backend_t backend = new ggml_backend {
        /* .guid   = */ &guid,
        /* .iface  = */ iface,
        /* .device = */ dev,
        /* .context = */ ctx
    };

    return backend;
}

static ggml_backend_buffer_type_t ttnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t buft = nullptr;

    if (!buft) {
        ttnn_buffer_type_context* ctx = new ttnn_buffer_type_context;
        ctx->device_id = (intptr_t)dev->context;
        snprintf(ctx->name, sizeof(ctx->name), "TTNN%d", ctx->device_id);

        ggml_backend_buffer_type_i iface = {};
        iface.get_name = ttnn_buffer_type_get_name;
        iface.alloc_buffer = ttnn_buffer_type_alloc_buffer;
        iface.get_alignment = ttnn_buffer_type_get_alignment;

        buft = new ggml_backend_buffer_type {
            /* .iface  = */ iface,
            /* .device = */ dev,
            /* .context = */ ctx
        };
    }

    return buft;
}

static bool ttnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
    UNUSED(dev);

    // Support basic ops for testing
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
            return true;
        default:
            return false;
    }
}

static bool ttnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    UNUSED(dev);
    UNUSED(buft);
    return false;
}

// Registry interface
static const char* ttnn_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_TTNN_NAME;
}

static size_t ttnn_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    if (!ttnn_init()) {
        return 0;
    }
    return g_devices.size();
}

static ggml_backend_dev_t ttnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    UNUSED(reg);

    static std::vector<ggml_backend_dev_t> devices;

    if (devices.empty()) {
        size_t count = ttnn_reg_get_device_count(reg);
        devices.reserve(count);

        for (size_t i = 0; i < count; i++) {
            ggml_backend_device_i iface = {};
            iface.get_name = ttnn_device_get_name;
            iface.get_description = ttnn_device_get_description;
            iface.get_memory = ttnn_device_get_memory;
            iface.get_type = ttnn_device_get_type;
            iface.get_props = ttnn_device_get_props;
            iface.init_backend = ttnn_device_init_backend;
            iface.get_buffer_type = ttnn_device_get_buffer_type;
            iface.supports_op = ttnn_device_supports_op;
            iface.supports_buft = ttnn_device_supports_buft;

            ggml_backend_dev_t dev = new ggml_backend_device {
                /* .iface   = */ iface,
                /* .reg     = */ reg,
                /* .context = */ (void*)(intptr_t)i
            };

            devices.push_back(dev);
        }
    }

    return (index < devices.size()) ? devices[index] : nullptr;
}

} // anonymous namespace

// Public API
extern "C" {

GGML_API ggml_backend_reg_t ggml_backend_ttnn_reg(void) {
    static ggml_backend_reg ttnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface = */ {
            /* .get_name = */ ttnn_reg_get_name,
            /* .get_device_count = */ ttnn_reg_get_device_count,
            /* .get_device = */ ttnn_reg_get_device,
            /* .get_proc_address = */ nullptr,
        },
        /* .context = */ nullptr
    };

    return &ttnn_reg;
}

GGML_API int ggml_backend_ttnn_get_device_count(void) {
    if (!ttnn_init()) {
        return 0;
    }
    return (int)g_devices.size();
}

GGML_API void ggml_backend_ttnn_get_device_description(int device, char* description, size_t description_size) {
    snprintf(description, description_size, "Tenstorrent TTNN stub device %d", device);
}

GGML_API ggml_backend_buffer_type_t ggml_backend_ttnn_buffer_type(int device) {
    UNUSED(device);
    return nullptr;
}

GGML_API ggml_backend_buffer_type_t ggml_backend_ttnn_host_buffer_type(void) {
    return nullptr;
}

GGML_API ggml_backend_buffer_type_t ggml_backend_ttnn_split_buffer_type(int main_device, const float* tensor_split) {
    UNUSED(main_device);
    UNUSED(tensor_split);
    return nullptr;
}

GGML_API bool ggml_backend_ttnn_register_host_buffer(void* buffer, size_t size) {
    UNUSED(buffer);
    UNUSED(size);
    return false;
}

GGML_API void ggml_backend_ttnn_unregister_host_buffer(void* buffer) {
    UNUSED(buffer);
}

} // extern "C"