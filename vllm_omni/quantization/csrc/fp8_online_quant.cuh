// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
//
// FP32/FP16/BF16 -> FP8 E4M3 online activation quantization primitives.
//
// This header is dependency-free (no Python/PyTorch) so the same source backs
// both the ahead-of-time build and the runtime-compiled extension.
//
// Numeric contract -- frozen against the kernel vLLM-Omni calls today, see
// docs/fp8_online_contract.md:
//
//   per-token:
//       amax      = max_j |x[r, j]|                       (fp32 comparison)
//       scale[r]  = (double)amax / 448.0, floored at 1/(448*512)
//                   then rounded to fp32
//       out[r, j] = e4m3_rn_sat( x[r, j] / scale[r] )
//
//   per-tensor:
//       scale[0]  = (double)amax / 448.0                  (no floor)
//
// The scale division is performed in double precision on purpose.  Computing it
// in fp32 changes ~50% of the scale values by 1 ULP, which in turn changes the
// quantized payload bytes; the reference implementation divides in double.
//
// Architecture coverage: guarded to sm_80 and newer, using only baseline PTX
// (`redux.sync.max.u32`, `cvt.rn.satfinite.e4m3x2.f32`).  The same source
// compiles for sm_80/86/89/90/100/120; sm_80/86 take the SDK converter branch.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace vllm_omni_fp8 {

__device__ __forceinline__ constexpr float fp8_e4m3_max() { return 448.0f; }
__device__ __forceinline__ constexpr double fp8_e4m3_max_d() { return 448.0; }

// 1 / (448 * 512): the smallest scale the reference implementation allows.
__device__ __forceinline__ constexpr double min_scale_d() {
  return 1.0 / (448.0 * 512.0);
}

// ---------------------------------------------------------------------------
// Packed E4M3 conversion.
//
// PTX puts the *first* source operand in the high 8 bits, so a low/high pair is
// assembled as cvt(dst, hi, lo).  The SDK fallback reproduces that ordering.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t pack_e4m3_pair(float lo, float hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint16_t packed;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(packed)
               : "f"(hi), "f"(lo));
  return packed;
#elif !defined(__CUDA_ARCH__)
  // Host compilation pass of this __device__-only helper: never reached, but it
  // must still parse, and the sm_80 converter names below only exist when a
  // real __CUDA_ARCH__ is selected.
  return 0;
#else
  // sm_80 / sm_86: no native FP8 pair conversion.  Two scalar conversions with
  // explicit round-to-nearest and saturation are equivalent per element, and the
  // pair is reassembled low-byte-first to match the PTX operand order.
  const uint16_t qlo = __nv_cvt_float_to_fp8(lo, __NV_SATFINITE, __NV_E4M3);
  const uint16_t qhi = __nv_cvt_float_to_fp8(hi, __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint16_t>(qlo | static_cast<uint16_t>(qhi << 8));
#endif
}

// Quantize one already-scaled value (saturating, round-to-nearest-even).
__device__ __forceinline__ uint8_t quant_e4m3(float v) {
  v = fminf(fmaxf(v, -fp8_e4m3_max()), fp8_e4m3_max());
  return static_cast<uint8_t>(pack_e4m3_pair(v, 0.0f) & 0xffu);
}

// ---------------------------------------------------------------------------
// Warp maximum of non-negative, non-NaN fp32 values.
//
// For non-negative floats the bit pattern is monotonic, so an unsigned integer
// maximum is a float maximum.  redux.sync is sm_80+; the shuffle form keeps the
// primitive total for targets the guard excludes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_max_nonneg(float v) {
  uint32_t bits = __float_as_uint(v);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t result;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
               : "=r"(result)
               : "r"(bits));
  return __uint_as_float(result);
#else
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, bits, delta);
    bits = bits > other ? bits : other;
  }
  return __uint_as_float(bits);
#endif
}

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Absolute maximum of VEC consecutive elements.
template <typename T, int VEC>
__device__ __forceinline__ float vec_absmax(const T* __restrict__ p) {
  float local = 0.0f;
  if constexpr (VEC == 4 && sizeof(T) == 2) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      local = fmaxf(local, fabsf(to_float<T>(v[i])));
    }
  } else if constexpr (VEC == 4 && sizeof(T) == 4) {
    const float4 raw = *reinterpret_cast<const float4*>(p);
    local = fmaxf(fmaxf(fabsf(raw.x), fabsf(raw.y)),
                  fmaxf(fabsf(raw.z), fabsf(raw.w)));
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      local = fmaxf(local, fabsf(to_float<T>(p[i])));
    }
  }
  return local;
}

// Load VEC consecutive elements as floats.
template <typename T, int VEC>
__device__ __forceinline__ void vec_load(const T* __restrict__ p,
                                         float (&out)[VEC]) {
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out[i] = to_float<T>(p[i]);
  }
}

// Store VEC quantized elements.  Only called with a fully in-range vector.
template <int VEC>
__device__ __forceinline__ void vec_store_fp8(uint8_t* __restrict__ p,
                                              const float* v) {
  if constexpr (VEC == 4) {
    const uint16_t lo = pack_e4m3_pair(v[0], v[1]);
    const uint16_t hi = pack_e4m3_pair(v[2], v[3]);
    *reinterpret_cast<uint32_t*>(p) =
        static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      p[i] = quant_e4m3(v[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Block amax -> scale, broadcast to every thread.
//
// `shared` must hold at least (blockDim.x + 31) / 32 floats.  Both barriers are
// unconditional: no thread may leave the collective early.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_scale_from_amax(float local,
                                                       float* shared,
                                                       const float* scale_ub,
                                                       bool has_scale_ub) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  const float amax = warp_max_nonneg(local);
  if (lane == 0) {
    shared[warp] = amax;
  }
  __syncthreads();

  if (warp == 0) {
    float block_amax = (lane < num_warps) ? shared[lane] : 0.0f;
    block_amax = warp_max_nonneg(block_amax);
    if (lane == 0) {
      if (has_scale_ub) {
        // Read on device so the kernel stays CUDA-graph capture safe: no host
        // round-trip is performed anywhere on this path.
        block_amax = fminf(block_amax, *scale_ub);
      }
      const double scale = static_cast<double>(block_amax) / fp8_e4m3_max_d();
      shared[0] =
          static_cast<float>(scale < min_scale_d() ? min_scale_d() : scale);
    }
  }
  __syncthreads();
  return shared[0];
}

// ---------------------------------------------------------------------------
// Scalar block amax + quantize, used when the row length is not a multiple of
// the vector width.  The host dispatch normally rejects such rows, but keeping
// the kernels total means a future caller cannot hit an out-of-bounds vector
// access by relaxing the gate.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void per_token_quant_row_scalar(
    const T* __restrict__ x_row, uint8_t* __restrict__ o_row, int64_t m,
    int64_t n, int64_t row, float* __restrict__ scale, float* smem,
    const float* scale_ub, bool has_scale_ub) {
  const int64_t tid = threadIdx.x;
  float local = 0.0f;
  for (int64_t i = tid; i < n; i += blockDim.x) {
    local = fmaxf(local, fabsf(to_float<T>(x_row[i])));
  }
  const float s = block_scale_from_amax(local, smem, scale_ub, has_scale_ub);
  if (tid == 0) {
    scale[row] = s;
  }
  for (int64_t i = tid; i < n; i += blockDim.x) {
    o_row[i] = quant_e4m3(to_float<T>(x_row[i]) / s);
  }
  (void)m;
}

// ---------------------------------------------------------------------------
// Per-token kernel: one CTA per row.
//
//   VEC        elements per packed vector (4 -> 8-byte load for 16-bit inputs)
//   HELD_VECS  vectors the thread keeps in registers so the row is read once;
//              0 selects the plain two-read form for rows that do not fit.
// ---------------------------------------------------------------------------
template <typename T, int VEC, int HELD_VECS>
__global__ void per_token_quant_kernel(const T* __restrict__ input, int64_t m,
                                       int64_t n, float* __restrict__ scale,
                                       uint8_t* __restrict__ output,
                                       const float* scale_ub,
                                       bool has_scale_ub) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ float smem[];

  const int64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }

  const T* __restrict__ x_row = input + row * n;
  uint8_t* __restrict__ o_row = output + row * n;

  const int64_t n_vec = n / VEC;      // whole VEC-wide vectors
  const int64_t vec_elems = n_vec * VEC;
  const int64_t tid = threadIdx.x;

  if (n % VEC != 0) {
    per_token_quant_row_scalar<T>(x_row, o_row, m, n, row, scale, smem, scale_ub,
                                 has_scale_ub);
    return;
  }

  float held[(HELD_VECS > 0 ? HELD_VECS : 1)][VEC];

  // The HELD form keeps HELD_VECS vectors per thread alive across the two
  // passes, so a row wider than blockDim.x * HELD_VECS vectors is covered by
  // repeating the register-resident chunk.  Only the first chunk is read once;
  // later chunks are re-read in pass 2, which is still cheaper than giving up
  // the register residency for the common case of a single chunk.
  const int64_t chunk_vecs =
      (HELD_VECS > 0) ? static_cast<int64_t>(blockDim.x) * HELD_VECS : 0;

  // ---- pass 1: amax ------------------------------------------------------
  float local = 0.0f;
  if constexpr (HELD_VECS > 0) {
    for (int64_t base = 0; base < n_vec; base += chunk_vecs) {
      for (int v = 0; v < HELD_VECS; ++v) {
        const int64_t idx = base + tid * HELD_VECS + v;
        if (idx < n_vec) {
          const T* p = x_row + idx * VEC;
          local = fmaxf(local, vec_absmax<T, VEC>(p));
          if (base == 0) {
            vec_load<T, VEC>(p, held[v]);
          }
        }
      }
    }
  } else {
    for (int64_t idx = tid; idx < n_vec; idx += blockDim.x) {
      local = fmaxf(local, vec_absmax<T, VEC>(x_row + idx * VEC));
    }
  }
  for (int64_t i = vec_elems + tid; i < n; i += blockDim.x) {
    local = fmaxf(local, fabsf(to_float<T>(x_row[i])));
  }

  const float s = block_scale_from_amax(local, smem, scale_ub, has_scale_ub);
  if (tid == 0) {
    scale[row] = s;
  }

  // ---- pass 2: quantize --------------------------------------------------
  if constexpr (HELD_VECS > 0) {
    for (int64_t base = 0; base < n_vec; base += chunk_vecs) {
      for (int v = 0; v < HELD_VECS; ++v) {
        const int64_t idx = base + tid * HELD_VECS + v;
        if (idx >= n_vec) {
          continue;
        }
        float y[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          y[i] = (base == 0 ? held[v][i] : to_float<T>(x_row[idx * VEC + i])) / s;
        }
        vec_store_fp8<VEC>(o_row + idx * VEC, y);
      }
    }
  } else {
    for (int64_t idx = tid; idx < n_vec; idx += blockDim.x) {
      float y[VEC];
      vec_load<T, VEC>(x_row + idx * VEC, y);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        y[i] = y[i] / s;
      }
      vec_store_fp8<VEC>(o_row + idx * VEC, y);
    }
  }
  for (int64_t i = vec_elems + tid; i < n; i += blockDim.x) {
    o_row[i] = quant_e4m3(to_float<T>(x_row[i]) / s);
  }
#endif
}

// ---------------------------------------------------------------------------
// Per-token kernel, warp-per-row: one warp owns one row.
//
// The block-per-row form above spends two CTA barriers per row and needs
// 256 threads per row.  For long sequences (large M) that geometry is poor: the
// grid becomes enormous while each CTA only covers one short row.  Giving each
// warp its own row removes every barrier -- the reduction is a single
// `redux.sync` inside the warp -- and lets one CTA cover WARPS rows.
// ---------------------------------------------------------------------------
template <typename T, int VEC, int WARPS_PER_CTA>
__global__ void per_token_quant_kernel_warp(const T* __restrict__ input,
                                            int64_t m, int64_t n,
                                            float* __restrict__ scale,
                                            uint8_t* __restrict__ output,
                                            const float* scale_ub,
                                            bool has_scale_ub) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WARPS_PER_CTA + warp;
  if (row >= m) {
    return;
  }

  const T* __restrict__ x_row = input + row * n;
  uint8_t* __restrict__ o_row = output + row * n;

  const int64_t n_vec = n / VEC;
  const int64_t vec_elems = n_vec * VEC;
  const int64_t stride = static_cast<int64_t>(32) * VEC;

  if (n % VEC != 0) {
    // Scalar fallback; a warp still owns the whole row, so the reduction is
    // one warp redux with no barrier.
    float sl = 0.0f;
    for (int64_t i = lane; i < n; i += 32) {
      sl = fmaxf(sl, fabsf(to_float<T>(x_row[i])));
    }
    float am = warp_max_nonneg(sl);
    if (has_scale_ub) {
      am = fminf(am, *scale_ub);
    }
    const double scd = static_cast<double>(am) / fp8_e4m3_max_d();
    const float ss = static_cast<float>(scd < min_scale_d() ? min_scale_d() : scd);
    if (lane == 0) {
      scale[row] = ss;
    }
    for (int64_t i = lane; i < n; i += 32) {
      o_row[i] = quant_e4m3(to_float<T>(x_row[i]) / ss);
    }
    return;
  }

  float local = 0.0f;
  for (int64_t off = static_cast<int64_t>(lane) * VEC; off < vec_elems; off += stride) {
    local = fmaxf(local, vec_absmax<T, VEC>(x_row + off));
  }
  for (int64_t i = vec_elems + lane; i < n; i += 32) {
    local = fmaxf(local, fabsf(to_float<T>(x_row[i])));
  }

  float amax = warp_max_nonneg(local);
  if (has_scale_ub) {
    amax = fminf(amax, *scale_ub);
  }
  const double sc = static_cast<double>(amax) / fp8_e4m3_max_d();
  const float s = static_cast<float>(sc < min_scale_d() ? min_scale_d() : sc);
  if (lane == 0) {
    scale[row] = s;
  }

  for (int64_t off = static_cast<int64_t>(lane) * VEC; off < vec_elems; off += stride) {
    float y[VEC];
    vec_load<T, VEC>(x_row + off, y);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      y[i] = y[i] / s;
    }
    vec_store_fp8<VEC>(o_row + off, y);
  }
  for (int64_t i = vec_elems + lane; i < n; i += 32) {
    o_row[i] = quant_e4m3(to_float<T>(x_row[i]) / s);
  }
#endif
}

// ---------------------------------------------------------------------------
// Per-tensor kernel: one CTA reduces the whole tensor, then quantizes it.
// ---------------------------------------------------------------------------
template <typename T, int VEC>
__global__ void per_tensor_quant_kernel(const T* __restrict__ input,
                                        int64_t numel,
                                        float* __restrict__ scale,
                                        uint8_t* __restrict__ output,
                                        const float* scale_ub,
                                        bool has_scale_ub) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ float smem[];
  const int64_t tid = threadIdx.x;
  const int64_t n_vec = numel / VEC;
  const int64_t vec_elems = n_vec * VEC;

  float local = 0.0f;
  for (int64_t idx = tid; idx < n_vec; idx += blockDim.x) {
    local = fmaxf(local, vec_absmax<T, VEC>(input + idx * VEC));
  }
  for (int64_t i = vec_elems + tid; i < numel; i += blockDim.x) {
    local = fmaxf(local, fabsf(to_float<T>(input[i])));
  }

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  const float amax = warp_max_nonneg(local);
  if (lane == 0) {
    smem[warp] = amax;
  }
  __syncthreads();

  if (warp == 0) {
    float block_amax = (lane < num_warps) ? smem[lane] : 0.0f;
    block_amax = warp_max_nonneg(block_amax);
    if (lane == 0) {
      if (has_scale_ub) {
        block_amax = fminf(block_amax, *scale_ub);
      }
      // No floor: the per-tensor path does not clamp to min_scale.
      smem[0] = static_cast<float>(static_cast<double>(block_amax) /
                                   fp8_e4m3_max_d());
    }
  }
  __syncthreads();

  const float s = smem[0];
  if (tid == 0) {
    scale[0] = s;
  }

  // The per-tensor reference multiplies by the reciprocal; it does NOT divide.
  // Measured over 9,437,184 payload bytes: x/s differs from the reference on 3
  // of 6 random tensors (1 byte each), x*(1/s) on none.  This is the opposite
  // choice from the per-token path, which really does divide.
  const float inv_s = 1.0f / s;

  for (int64_t idx = tid; idx < n_vec; idx += blockDim.x) {
    float y[VEC];
    vec_load<T, VEC>(input + idx * VEC, y);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      y[i] = y[i] * inv_s;
    }
    vec_store_fp8<VEC>(output + idx * VEC, y);
  }
  for (int64_t i = vec_elems + tid; i < numel; i += blockDim.x) {
    output[i] = quant_e4m3(to_float<T>(input[i]) * inv_s);
  }
#endif
}

// ---------------------------------------------------------------------------
// Per-tensor, multi-block form.
//
// The single-CTA kernel above is only viable while one block can cover the
// tensor; at [128, 4096] it measured 0.164x and at [512, 3072] 0.046x against
// the reference, because 256 threads serialise over the whole tensor.  For
// anything larger the work is split in two passes:
//
//   pass 1  grid-strided partial amax -> atomicMax into a global uint32
//   pass 2  read that amax, form the scale, and quantize grid-strided
//
// The amax is non-negative, so its IEEE bit pattern is monotonic and an
// unsigned atomicMax is a float max.  Pass 2 recomputes the scale in every
// block (one double divide per block, negligible) so no third launch or grid
// barrier is needed; block 0 thread 0 is the single writer of `scale`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_amax_only(float local, float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  const float amax = warp_max_nonneg(local);
  if (lane == 0) {
    shared[warp] = amax;
  }
  __syncthreads();

  float block_amax = 0.0f;
  if (warp == 0) {
    block_amax = (lane < num_warps) ? shared[lane] : 0.0f;
    block_amax = warp_max_nonneg(block_amax);
  }
  return block_amax;
}

template <typename T, int VEC>
__global__ void per_tensor_amax_kernel(const T* __restrict__ input,
                                       int64_t numel,
                                       unsigned* __restrict__ amax_bits) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ float smem[];
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t n_vec = numel / VEC;
  const int64_t vec_elems = n_vec * VEC;

  float local = 0.0f;
  for (int64_t idx = tid; idx < n_vec; idx += stride) {
    local = fmaxf(local, vec_absmax<T, VEC>(input + idx * VEC));
  }
  for (int64_t i = vec_elems + tid; i < numel; i += stride) {
    local = fmaxf(local, fabsf(to_float<T>(input[i])));
  }

  const float block_amax = block_amax_only(local, smem);
  if (threadIdx.x == 0) {
    atomicMax(amax_bits, __float_as_uint(block_amax));
  }
#endif
}

template <typename T, int VEC>
__global__ void per_tensor_quant_from_amax_kernel(
    const T* __restrict__ input, int64_t numel,
    const unsigned* __restrict__ amax_bits, float* __restrict__ scale,
    uint8_t* __restrict__ output, const float* scale_ub, bool has_scale_ub) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  float amax = __uint_as_float(*amax_bits);
  if (has_scale_ub) {
    amax = fminf(amax, *scale_ub);
  }
  // No floor: the per-tensor path does not clamp to min_scale.
  const float s = static_cast<float>(static_cast<double>(amax) / fp8_e4m3_max_d());
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    scale[0] = s;
  }
  // Reciprocal multiply, matching the reference; see the single-CTA kernel.
  const float inv_s = 1.0f / s;

  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t n_vec = numel / VEC;
  const int64_t vec_elems = n_vec * VEC;

  for (int64_t idx = tid; idx < n_vec; idx += stride) {
    float y[VEC];
    vec_load<T, VEC>(input + idx * VEC, y);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      y[i] = y[i] * inv_s;
    }
    vec_store_fp8<VEC>(output + idx * VEC, y);
  }
  for (int64_t i = vec_elems + tid; i < numel; i += stride) {
    output[i] = quant_e4m3(to_float<T>(input[i]) * inv_s);
  }
#endif
}

}  // namespace vllm_omni_fp8
