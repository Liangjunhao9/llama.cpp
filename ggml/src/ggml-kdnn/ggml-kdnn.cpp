#include "ggml-kdnn.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <kdnn.hpp>
#include <service/kdnn_threading.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>

static void ggml_kdnn_fp16_to_fp32(const ggml_fp16_t * src, float * dst, int64_t n) {
#if defined(__aarch64__)
    using fp16_alias = __fp16 __attribute__((may_alias));
    const auto * values = reinterpret_cast<const fp16_alias *>(src);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = values[i];
    }
#else
    ggml_fp16_to_fp32_row(src, dst, n);
#endif
}

struct ggml_backend_kdnn_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<float[]> src0_f32;
    size_t src0_f32_elements = 0;
    std::map<std::tuple<int64_t, int64_t, int64_t, ggml_type, int>, std::unique_ptr<KDNN::Gemm>> gemms;
};

static KDNN::TensorInfo kdnn_info(KDNN::Shape shape, KDNN::Element::TypeT type, KDNN::Layout layout) {
    return KDNN::TensorInfo(shape, type, layout);
}

static void ggml_backend_kdnn_mul_mat(ggml_backend_kdnn_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    const size_t src0_plane_elements = (size_t) ne01 * (size_t) ne00;
    if (src0->type == GGML_TYPE_F16 && ctx->src0_f32_elements < src0_plane_elements) {
        ctx->src0_f32.reset(new float[src0_plane_elements]);
        ctx->src0_f32_elements = src0_plane_elements;
    }

    KDNN::Threading::SetNumThreadsLocal(ctx->n_threads);
    // ggml's F16-weight MUL_MAT contract is F16 weights x F32 activations -> F32 output.
    // Convert the weights to F32 and keep activation/output in F32 so that KDNN follows
    // the same numerical path as the ggml CPU and BLAS backends.
    const auto type = KDNN::Element::TypeT::F32;
    const auto a_info = kdnn_info(KDNN::Shape { (size_t) ne11, (size_t) ne10 }, type, KDNN::Layout::AB);
    const auto b_info = kdnn_info(KDNN::Shape { (size_t) ne00, (size_t) ne01 }, type, KDNN::Layout::BA);
    const auto c_info = kdnn_info(KDNN::Shape { (size_t) ne1,  (size_t) ne0  }, type, KDNN::Layout::AB);
    const auto key = std::make_tuple(ne11, ne01, ne10, src0->type, ctx->n_threads);
    auto & gemm = ctx->gemms[key];
    if (!gemm) {
        gemm = std::make_unique<KDNN::Gemm>(a_info, b_info, c_info);
    }
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;
            const float * src1_plane = (const float *) ((const char *) src1->data + i12 * nb12 + i13 * nb13);
            const void * a = src1_plane;
            const void * b = (const char *) src0->data + i02 * nb02 + i03 * nb03;
            void * c = (char *) dst->data + i12 * nb2 + i13 * nb3;
            if (src0->type == GGML_TYPE_F16) {
                ggml_kdnn_fp16_to_fp32((const ggml_fp16_t *) b, ctx->src0_f32.get(), src0_plane_elements);
                b = ctx->src0_f32.get();
            }
            gemm->Run(a, b, c);
        }
    }
}

static const char * ggml_backend_kdnn_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "KDNN";
}

static void ggml_backend_kdnn_free(ggml_backend_t backend) {
    delete (ggml_backend_kdnn_context *) backend->context;
    delete backend;
}

static ggml_status ggml_backend_kdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_kdnn_context *) backend->context;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_kdnn_mul_mat(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i kdnn_backend_i = {
    ggml_backend_kdnn_get_name, ggml_backend_kdnn_free,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, ggml_backend_kdnn_graph_compute,
    nullptr, nullptr, nullptr,
};

static ggml_guid_t ggml_backend_kdnn_guid(void) {
    static ggml_guid guid = { 0x4b, 0x44, 0x4e, 0x4e, 0x91, 0x66, 0x45, 0xa1, 0x83, 0xd2, 0x60, 0x78, 0x72, 0x44, 0x31, 0x10 };
    return &guid;
}

ggml_backend_t ggml_backend_kdnn_init(void) {
    return new ggml_backend {
        ggml_backend_kdnn_guid(), kdnn_backend_i,
        ggml_backend_reg_dev_get(ggml_backend_kdnn_reg(), 0),
        new ggml_backend_kdnn_context,
    };
}

bool ggml_backend_is_kdnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_kdnn_guid());
}

void ggml_backend_kdnn_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_kdnn(backend));
    const char * value = std::getenv("GGML_KDNN_N_THREADS");
    const int override_threads = value != nullptr ? std::atoi(value) : 0;
    ((ggml_backend_kdnn_context *) backend->context)->n_threads = override_threads > 0 ? override_threads : n_threads;
}

static const char * ggml_backend_kdnn_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "KDNN";
}

static const char * ggml_backend_kdnn_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Kunpeng DNN Library";
}

static void ggml_backend_kdnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_kdnn_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_kdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_kdnn_device_get_name(dev);
    props->description = ggml_backend_kdnn_device_get_description(dev);
    props->type = ggml_backend_kdnn_device_get_type(dev);
    ggml_backend_kdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { false, false, true, false, true };
}

static ggml_backend_t ggml_backend_kdnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_kdnn_init();
}

static ggml_backend_buffer_type_t ggml_backend_kdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_kdnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_kdnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];
            return ggml_get_op_params_i32(op, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
                   (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                   src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                   op->ne[0] >= 32 && src1->ne[0] >= 32;
        }
        default:
            return false;
    }
}

static bool ggml_backend_kdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i kdnn_device_i = {
    ggml_backend_kdnn_device_get_name, ggml_backend_kdnn_device_get_description,
    ggml_backend_kdnn_device_get_memory, ggml_backend_kdnn_device_get_type,
    ggml_backend_kdnn_device_get_props, ggml_backend_kdnn_device_init_backend,
    ggml_backend_kdnn_device_get_buffer_type, nullptr,
    ggml_backend_kdnn_device_buffer_from_host_ptr, ggml_backend_kdnn_device_supports_op,
    ggml_backend_kdnn_device_supports_buft, nullptr,
    nullptr, nullptr, nullptr,
};

static const char * ggml_backend_kdnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "KDNN";
}

static size_t ggml_backend_kdnn_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_kdnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = { kdnn_device_i, reg, nullptr };
    return &device;
}

static void * ggml_backend_kdnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_kdnn_set_n_threads;
    }
    return nullptr;
}

static const ggml_backend_reg_i kdnn_reg_i = {
    ggml_backend_kdnn_reg_get_name,
    ggml_backend_kdnn_reg_get_device_count,
    ggml_backend_kdnn_reg_get_device,
    ggml_backend_kdnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_kdnn_reg(void) {
    static ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, kdnn_reg_i, nullptr };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_kdnn_reg)
