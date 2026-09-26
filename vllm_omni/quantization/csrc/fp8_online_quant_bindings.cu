// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
//
// Host-side launchers + pybind11 bindings for the FP8 online quantization
// kernels in fp8_online_quant.cuh.
//
// Two entry points are exposed:
//   fp8_online_per_token_quant(out, inp, scale, scale_ub_or_undefined)
//   fp8_online_per_tensor_quant(out, inp, scale, scale_ub_or_undefined)
//
// Both write into caller-allocated `out` (uint8/fp8_e4m3) and `scale` (fp32).
// A "not handled" return value is never used: any input the fast path does not
// cover raises, so the Python dispatch layer is the single place that decides
// between this kernel and the reference implementation.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

#include "fp8_online_quant.cuh"

namespace {

using vllm_omni_fp8::per_tensor_amax_kernel;
using vllm_omni_fp8::per_tensor_quant_from_amax_kernel;
using vllm_omni_fp8::per_tensor_quant_kernel;
using vllm_omni_fp8::per_token_quant_kernel;
using vllm_omni_fp8::per_token_quant_kernel_warp;

constexpr int kBlockThreads = 256;
constexpr int kMaxHeldVecs = 4;

// The block-per-row mapping spends two CTA barriers per row and one CTA per row.
// It keeps the input in registers and issues a single kernel, which wins for
// every row count the Python gate accepts.  Measured, not guessed: the
// warp-per-row mapping was never observed to beat it below M=4096, so the switch
// sits at the top of the supported envelope.  The Python dispatch gate
// (FASTPATH_MAX_ROWS) is the authority on which inputs are accepted at all; this
// constant only chooses between the two mappings inside that envelope.
constexpr int64_t kDefaultBlockPerRowMaxRows = 512;

// Per-tensor crossover between the single-CTA and multi-block forms, and the
// cap on blocks for the multi-block form.  Measured; see
// docs/fp8_online_results.md.
constexpr int64_t kPerTensorSingleCtaMaxElems = 32768;
constexpr int64_t kPerTensorMaxBlocks = 1024;

// Read once, from the environment, so the mapping crossover can be re-measured
// without editing and rebuilding.  VLLM_OMNI_FP8_ROW_MAPPING selects the
// strategy explicitly: "block" or "warp".
struct MappingConfig {
  int64_t block_per_row_max_rows = kDefaultBlockPerRowMaxRows;
  int force = 0;  // 0 = automatic, 1 = force block-per-row, 2 = force warp-per-row

  static const MappingConfig& get() {
    static const MappingConfig cfg = [] {
      MappingConfig c;
      if (const char* s = std::getenv("VLLM_OMNI_FP8_ROW_MAPPING")) {
        const std::string v(s);
        if (v == "block") {
          c.force = 1;
        } else if (v == "warp") {
          c.force = 2;
        }
      }
      if (const char* s = std::getenv("VLLM_OMNI_FP8_BLOCK_ROW_MAX")) {
        c.block_per_row_max_rows = std::atoll(s);
      }
      return c;
    }();
    return cfg;
  }
};

// Rows handled by one CTA in the warp-per-row mapping (8 warps x 1 row).
constexpr int kWarpRowsPerCta = kBlockThreads / 32;

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Shared memory needed for the block reduction (one float per warp).
size_t reduce_smem_bytes() {
  return static_cast<size_t>(ceil_div(kBlockThreads, 32)) * sizeof(float);
}

// ---------------------------------------------------------------------------
// Type dispatch for the element type of the input tensor.
// ---------------------------------------------------------------------------

// CUDA element type backing each supported torch dtype.
template <typename T>
struct CudaElem;
template <>
struct CudaElem<float> {
  using type = float;
};
template <>
struct CudaElem<at::Half> {
  using type = __half;
};
template <>
struct CudaElem<at::BFloat16> {
  using type = __nv_bfloat16;
};

// Dispatch on the input element type, binding `input_t` for the callable,
// which is invoked as launch.template operator()<input_t>().
//
// Only fp32/fp16/bf16 are supported, so this is a plain if-constexpr chain
// rather than AT_DISPATCH_FLOATING_TYPES (which would also instantiate for
// double, a type the kernels do not accept).
template <typename Fn>
void dispatch_input_type(const torch::Tensor& inp, Fn&& launch) {
  const auto dt = inp.scalar_type();
  if (dt == at::ScalarType::Float) {
    launch.template operator()<float>();
  } else if (dt == at::ScalarType::Half) {
    launch.template operator()<__half>();
  } else if (dt == at::ScalarType::BFloat16) {
    launch.template operator()<__nv_bfloat16>();
  } else {
    TORCH_CHECK(false, "fp8_online_quant: unsupported input dtype ", dt);
  }
}

// Block-per-row mapping (input kept in registers, one CTA per row).
template <int VEC, int HELD_VECS>
void launch_per_token(const torch::Tensor& inp, torch::Tensor& out,
                      torch::Tensor& scale, const float* scale_ub,
                      bool has_scale_ub, int64_t m, int64_t n,
                      cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(m));
  const dim3 block(kBlockThreads);
  const size_t smem = reduce_smem_bytes();

  dispatch_input_type(inp, [&]<typename input_t>() {
    per_token_quant_kernel<input_t, VEC, HELD_VECS><<<grid, block, smem, stream>>>(
        reinterpret_cast<const input_t*>(inp.const_data_ptr()), m, n,
        scale.data_ptr<float>(), reinterpret_cast<uint8_t*>(out.data_ptr()),
        scale_ub, has_scale_ub);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Warp-per-row mapping (no barriers; better for long sequences).
template <int VEC>
void launch_per_token_warp(const torch::Tensor& inp, torch::Tensor& out,
                           torch::Tensor& scale, const float* scale_ub,
                           bool has_scale_ub, int64_t m, int64_t n,
                           cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(ceil_div(m, kWarpRowsPerCta)));
  const dim3 block(kBlockThreads);

  dispatch_input_type(inp, [&]<typename input_t>() {
    per_token_quant_kernel_warp<input_t, VEC, kWarpRowsPerCta>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const input_t*>(inp.const_data_ptr()), m, n,
            scale.data_ptr<float>(), reinterpret_cast<uint8_t*>(out.data_ptr()),
            scale_ub, has_scale_ub);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int VEC>
void launch_per_tensor(const torch::Tensor& inp, torch::Tensor& out,
                       torch::Tensor& scale, const float* scale_ub,
                       bool has_scale_ub, int64_t numel, cudaStream_t stream) {
  const dim3 grid(1);
  const dim3 block(kBlockThreads);
  const size_t smem = reduce_smem_bytes();

  dispatch_input_type(inp, [&]<typename input_t>() {
    per_tensor_quant_kernel<input_t, VEC><<<grid, block, smem, stream>>>(
        reinterpret_cast<const input_t*>(inp.const_data_ptr()), numel,
        scale.data_ptr<float>(), reinterpret_cast<uint8_t*>(out.data_ptr()),
        scale_ub, has_scale_ub);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Multi-block per-tensor: partial amax (atomicMax) then a grid-strided quantize.
// `scratch` holds one uint32 and must be zeroed before the amax pass.
template <int VEC>
void launch_per_tensor_multiblock(const torch::Tensor& inp, torch::Tensor& out,
                                  torch::Tensor& scale, torch::Tensor& scratch,
                                  const float* scale_ub, bool has_scale_ub,
                                  int64_t numel, int blocks, cudaStream_t stream) {
  const size_t smem = reduce_smem_bytes();
  unsigned* amax_bits = reinterpret_cast<unsigned*>(scratch.data_ptr<int32_t>());

  dispatch_input_type(inp, [&]<typename input_t>() {
    per_tensor_amax_kernel<input_t, VEC><<<blocks, kBlockThreads, smem, stream>>>(
        reinterpret_cast<const input_t*>(inp.const_data_ptr()), numel, amax_bits);
    per_tensor_quant_from_amax_kernel<input_t, VEC>
        <<<blocks, kBlockThreads, 0, stream>>>(
            reinterpret_cast<const input_t*>(inp.const_data_ptr()), numel,
            amax_bits, scale.data_ptr<float>(),
            reinterpret_cast<uint8_t*>(out.data_ptr()), scale_ub, has_scale_ub);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Shared argument validation.
// ---------------------------------------------------------------------------

void check_common(const torch::Tensor& inp, const torch::Tensor& out,
                  const torch::Tensor& scale) {
  TORCH_CHECK(inp.is_cuda(), "fp8_online_quant: input must be a CUDA tensor");
  TORCH_CHECK(inp.is_contiguous(),
              "fp8_online_quant: input must be contiguous");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous(),
              "fp8_online_quant: output must be a contiguous CUDA tensor");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_online_quant: output must be float8_e4m3fn");
  TORCH_CHECK(out.numel() == inp.numel(),
              "fp8_online_quant: output must match the input element count");
  TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() &&
                  scale.scalar_type() == at::ScalarType::Float,
              "fp8_online_quant: scale must be a contiguous fp32 CUDA tensor");
  TORCH_CHECK(inp.scalar_type() == at::ScalarType::Float ||
                  inp.scalar_type() == at::ScalarType::Half ||
                  inp.scalar_type() == at::ScalarType::BFloat16,
              "fp8_online_quant: input must be fp32, fp16 or bf16");
}

// Returns a device pointer (or nullptr).  The value is read inside the kernel
// so that no host synchronization happens on the fast path -- this keeps the
// operator safe to capture in a CUDA graph.
const float* resolve_scale_ub(const std::optional<torch::Tensor>& scale_ub,
                              bool& has_scale_ub) {
  has_scale_ub = scale_ub.has_value();
  if (!has_scale_ub) {
    return nullptr;
  }
  const torch::Tensor& ub = *scale_ub;
  TORCH_CHECK(ub.is_cuda() && ub.numel() == 1 && ub.is_contiguous(),
              "fp8_online_quant: scale_ub must be a one-element contiguous "
              "CUDA tensor");
  TORCH_CHECK(ub.scalar_type() == at::ScalarType::Float,
              "fp8_online_quant: scale_ub must be float32");
  return ub.const_data_ptr<float>();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

void fp8_online_per_token_quant(torch::Tensor out, torch::Tensor inp,
                                torch::Tensor scale,
                                std::optional<torch::Tensor> scale_ub) {
  check_common(inp, out, scale);
  TORCH_CHECK(inp.dim() == 2, "fp8_online_per_token_quant: input must be 2D");

  const int64_t m = inp.size(0);
  const int64_t n = inp.size(1);
  TORCH_CHECK(scale.numel() == m,
              "fp8_online_per_token_quant: scale must have one entry per row");
  if (m == 0 || n == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard guard(inp.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool has_scale_ub = false;
  const float* ub = resolve_scale_ub(scale_ub, has_scale_ub);

  // Row length must be a multiple of 4 so the vector body covers every element
  // and the scalar tail is provably empty.  Other rows are rejected here and
  // handled by the Python dispatch layer, which keeps the fast path auditable.
  TORCH_CHECK(n % 4 == 0,
              "fp8_online_per_token_quant: row length must be a multiple of 4");
  const int64_t n_vec = n / 4;

  const MappingConfig& map = MappingConfig::get();
  const bool use_warp =
      (map.force == 2) || (map.force == 0 && m > map.block_per_row_max_rows);

  // Long sequences: one warp per row, no barriers, several rows per CTA.
  if (use_warp) {
    launch_per_token_warp<4>(inp, out, scale, ub, has_scale_ub, m, n, stream);
    return;
  }

  // Short sequences: one CTA per row with the row held in registers.
  if (n_vec <= static_cast<int64_t>(kBlockThreads) * kMaxHeldVecs &&
      n_vec % kBlockThreads == 0) {
    const int64_t held = n_vec / kBlockThreads;
    switch (held) {
      case 1:
        launch_per_token<4, 1>(inp, out, scale, ub, has_scale_ub, m, n, stream);
        return;
      case 2:
        launch_per_token<4, 2>(inp, out, scale, ub, has_scale_ub, m, n, stream);
        return;
      case 3:
        launch_per_token<4, 3>(inp, out, scale, ub, has_scale_ub, m, n, stream);
        return;
      case 4:
        launch_per_token<4, 4>(inp, out, scale, ub, has_scale_ub, m, n, stream);
        return;
      default:
        break;
    }
  }
  // Row does not fit the register-resident mapping: stream it twice.
  launch_per_token<4, 0>(inp, out, scale, ub, has_scale_ub, m, n, stream);
}

void fp8_online_per_tensor_quant(torch::Tensor out, torch::Tensor inp,
                                 torch::Tensor scale,
                                 std::optional<torch::Tensor> scale_ub) {
  check_common(inp, out, scale);
  TORCH_CHECK(scale.numel() == 1,
              "fp8_online_per_tensor_quant: scale must have one element");
  const int64_t numel = inp.numel();
  if (numel == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard guard(inp.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool has_scale_ub = false;
  const float* ub = resolve_scale_ub(scale_ub, has_scale_ub);

  const int vec = (numel % 4 == 0) ? 4 : 1;

  // One CTA is only viable while it can cover the tensor; the crossover was
  // measured, not guessed.  Below it the single launch wins; above it the
  // multi-block two-pass wins by a wide margin.
  if (numel <= kPerTensorSingleCtaMaxElems) {
    if (vec == 4) {
      launch_per_tensor<4>(inp, out, scale, ub, has_scale_ub, numel, stream);
    } else {
      launch_per_tensor<1>(inp, out, scale, ub, has_scale_ub, numel, stream);
    }
    return;
  }

  // Zeroing the amax accumulator is part of the algorithm, not an extra copy
  // the caller owns: one uint32, on the caller's stream.
  torch::Tensor scratch = at::zeros(
      {1}, inp.options().dtype(at::kInt).device(inp.device()));
  const int blocks = static_cast<int>(
      std::min<int64_t>(kPerTensorMaxBlocks,
                        ceil_div(numel, static_cast<int64_t>(kBlockThreads) * 8)));
  if (vec == 4) {
    launch_per_tensor_multiblock<4>(inp, out, scale, scratch, ub, has_scale_ub,
                                    numel, blocks, stream);
  } else {
    launch_per_tensor_multiblock<1>(inp, out, scale, scratch, ub, has_scale_ub,
                                    numel, blocks, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "vLLM-Omni FP8 online activation quantization (sm_80+)";
  m.def("fp8_online_per_token_quant", &fp8_online_per_token_quant,
        "Dynamic per-token FP8 E4M3 quantization (CUDA)");
  m.def("fp8_online_per_tensor_quant", &fp8_online_per_tensor_quant,
        "Dynamic per-tensor FP8 E4M3 quantization (CUDA)");
}
