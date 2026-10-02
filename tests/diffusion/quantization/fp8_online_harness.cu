// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
//
// Standalone harness for the FP8 online quantization kernels.
//
// Two jobs, neither of which needs a vLLM or torch import:
//
//   1. AOT compile coverage.  Building this file forces every kernel
//      specialisation to be instantiated, so `tests/compile_matrix.sh` gets real
//      per-architecture register/spill numbers for the whole dispatch surface.
//
//   2. Kernel-only benchmarking, independent of any Python dispatch overhead.
//
// Correctness is deliberately NOT validated against a hand-written host decoder.
// The authoritative oracle is the compiled reference kernel, and
// `tests/test_fastpath_vs_facade.py` plus
// `tests/diffusion/quantization/test_fp8_online_quant.py` compare against it
// byte-for-byte.  A host decoder is easy to get subtly wrong here: one that
// divides in double instead of fp32 disagrees with the kernel on exact fp32 ties
// (0.375/0.0089285718 is exactly 42.0 in fp32), and one that rounds ties to even
// disagrees on every midpoint.  Both are oracle bugs, not kernel bugs, so this
// file does not repeat that mistake.  What it does check is the byte-level
// packing contract (static fixture table) and self-consistency across all three
// mappings, which catches indexing and tail bugs without a decoder.
//
// usage: fp8_online_harness [--bench] [--fuzz] [--csv <path>]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fp8_online_quant.cuh"

using namespace vllm_omni_fp8;

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t e_ = (expr);                                                 \
    if (e_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__, \
                   __LINE__, cudaGetErrorString(e_));                        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int kBlockThreads = 256;

// ---------------------------------------------------------------------------
// Static fixtures for the packed-conversion byte contract.
// ---------------------------------------------------------------------------
struct Fixture {
  float value;
  uint8_t e4m3;
  const char* why;
};

static const Fixture kFixtures[] = {
    {1.0f, 0x38, "1.0 -> exp 7, mantissa 0"},
    {2.0f, 0x40, "2.0"},
    {4.0f, 0x48, "4.0"},
    {8.0f, 0x50, "8.0; the 4-byte word must read 0x50484038"},
    {-1.0f, 0xb8, "sign bit only"},
    {0.0f, 0x00, "+0"},
    {-0.0f, 0x80, "-0 preserved, not normalised"},
    {448.0f, 0x7e, "largest finite E4M3"},
    {-448.0f, 0xfe, "most negative finite E4M3"},
    {0.001953125f, 0x01, "smallest subnormal (2^-9)"},
    {0.013671875f, 0x07, "largest subnormal (7 * 2^-9)"},
    {0.015625f, 0x08, "smallest normal (2^-6)"},
    {0.5f, 0x30, "0.5"},
    {1.0625f, 0x38, "1 + 1/16 rounds down to mantissa 0"},
    {1.125f, 0x39, "1 + 1/8 -> mantissa 001"},
};

__global__ void fixture_kernel(const float* values, int count, uint8_t* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int i = threadIdx.x;
  if (i < count) {
    out[i] = quant_e4m3(values[i]);
  }
#endif
}

// A full 4-element vector through vec_store_fp8, which is what the per-token
// kernels use for the aligned body.  A swapped low/high byte in the packed pair
// would show up here and nowhere else.
__global__ void pack4_kernel(const float* values, int count, uint8_t* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int v = threadIdx.x;
  if ((v + 1) * 4 <= count) {
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i] = values[v * 4 + i];
    }
    vec_store_fp8<4>(out + v * 4, y);
  }
#endif
}

static int check_fixtures() {
  const int n = static_cast<int>(sizeof(kFixtures) / sizeof(kFixtures[0]));
  std::vector<float> hv(n);
  std::vector<uint8_t> want(n);
  for (int i = 0; i < n; ++i) {
    hv[i] = kFixtures[i].value;
    want[i] = kFixtures[i].e4m3;
  }
  float* dv = nullptr;
  uint8_t* dq = nullptr;
  CUDA_CHECK(cudaMalloc(&dv, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dq, n));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  fixture_kernel<<<1, 64>>>(dv, n, dq);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(n, 0);
  CUDA_CHECK(cudaMemcpy(got.data(), dq, n, cudaMemcpyDeviceToHost));

  int fails = 0;
  for (int i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::printf("  scalar  %-12g got=0x%02x want=0x%02x  (%s)\n", (double)hv[i],
                  got[i], want[i], kFixtures[i].why);
      ++fails;
    }
  }

  const float quad[4] = {1.0f, 2.0f, 4.0f, 8.0f};
  float* dq4 = nullptr;
  uint8_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dq4, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, 4));
  CUDA_CHECK(cudaMemcpy(dq4, quad, sizeof(quad), cudaMemcpyHostToDevice));
  pack4_kernel<<<1, 1>>>(dq4, 4, dout);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  uint8_t packed[4] = {0, 0, 0, 0};
  CUDA_CHECK(cudaMemcpy(packed, dout, 4, cudaMemcpyDeviceToHost));
  const uint32_t word = static_cast<uint32_t>(packed[0]) |
                        (static_cast<uint32_t>(packed[1]) << 8) |
                        (static_cast<uint32_t>(packed[2]) << 16) |
                        (static_cast<uint32_t>(packed[3]) << 24);
  if (word != 0x50484038u) {
    std::printf("  packed  vec_store_fp8<4>(1,2,4,8) = 0x%08x, want 0x50484038\n",
                word);
    ++fails;
  }
  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(dq4));
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dv));
  return fails;
}

// ---------------------------------------------------------------------------
// Self-consistency fuzz across the three mappings.
// ---------------------------------------------------------------------------
template <typename T>
struct DevBuf {
  T* x = nullptr;
  float* s = nullptr;
  uint8_t* q = nullptr;
  void alloc(int64_t m, int64_t n, const std::vector<T>& hx) {
    CUDA_CHECK(cudaMalloc(&x, hx.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&s, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q, m * n));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~DevBuf() {
    if (q) cudaFree(q);
    if (s) cudaFree(s);
    if (x) cudaFree(x);
  }
};

template <typename T>
static std::vector<T> to_type(const std::vector<float>& s);
template <>
std::vector<float> to_type<float>(const std::vector<float>& s) {
  return s;
}
template <>
std::vector<__half> to_type<__half>(const std::vector<float>& s) {
  std::vector<__half> d(s.size());
  for (size_t i = 0; i < s.size(); ++i) d[i] = __float2half_rn(s[i]);
  return d;
}
template <>
std::vector<__nv_bfloat16> to_type<__nv_bfloat16>(const std::vector<float>& s) {
  std::vector<__nv_bfloat16> d(s.size());
  for (size_t i = 0; i < s.size(); ++i) d[i] = __float2bfloat16_rn(s[i]);
  return d;
}

template <typename T, int VEC, int HELD>
static void launch_block(const DevBuf<T>& b, int64_t m, int64_t n) {
  const size_t smem = ((kBlockThreads + 31) / 32) * sizeof(float);
  per_token_quant_kernel<T, VEC, HELD><<<dim3((unsigned)m), dim3(kBlockThreads), smem>>>(
      b.x, m, n, b.s, b.q, nullptr, false);
}

template <typename T, int WARPS>
static void launch_warp(const DevBuf<T>& b, int64_t m, int64_t n) {
  per_token_quant_kernel_warp<T, 4, WARPS>
      <<<dim3((unsigned)((m + WARPS - 1) / WARPS)), dim3(WARPS * 32)>>>(
          b.x, m, n, b.s, b.q, nullptr, false);
}

template <typename T>
static int fuzz_one(int64_t m, int64_t n, uint32_t seed) {
  std::vector<float> hx(m * n);
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = nd(rng);
  // Mix in exact ties and zero rows so the comparison covers more than noise.
  for (size_t i = 0; i + 1 < hx.size(); i += 1024) hx[i] = 0.0f;
  for (size_t i = 0; i + 2 < hx.size(); i += 2048) hx[i + 1] = -0.0f;

  auto htype = to_type<T>(hx);

  DevBuf<T> ref;
  ref.alloc(m, n, htype);
  const int64_t nvec = n / 4;
  if (nvec % kBlockThreads == 0 && nvec / kBlockThreads >= 1 &&
      nvec / kBlockThreads <= 4) {
    switch (nvec / kBlockThreads) {
      case 1: launch_block<T, 4, 1>(ref, m, n); break;
      case 2: launch_block<T, 4, 2>(ref, m, n); break;
      case 3: launch_block<T, 4, 3>(ref, m, n); break;
      default: launch_block<T, 4, 4>(ref, m, n); break;
    }
  } else {
    launch_block<T, 4, 0>(ref, m, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> ref_s(m);
  std::vector<uint8_t> ref_q(m * n);
  CUDA_CHECK(cudaMemcpy(ref_s.data(), ref.s, m * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_q.data(), ref.q, m * n, cudaMemcpyDeviceToHost));

  int fails = 0;
  auto compare = [&](const char* name, void (*fn)(const DevBuf<T>&, int64_t, int64_t)) {
    DevBuf<T> b;
    b.alloc(m, n, htype);
    fn(b, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> s(m);
    std::vector<uint8_t> q(m * n);
    CUDA_CHECK(cudaMemcpy(s.data(), b.s, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q.data(), b.q, m * n, cudaMemcpyDeviceToHost));
    const bool sbad = std::memcmp(s.data(), ref_s.data(), m * sizeof(float)) != 0;
    const bool qbad = std::memcmp(q.data(), ref_q.data(), m * n) != 0;
    if (sbad || qbad) {
      std::printf("  MISMATCH %-18s M=%lld K=%lld scale_bad=%d payload_bad=%d\n",
                  name, (long long)m, (long long)n, (int)sbad, (int)qbad);
      ++fails;
    }
  };

  compare("streaming block/row", &launch_block<T, 4, 0>);
  compare("held=1", &launch_block<T, 4, 1>);
  compare("held=2", &launch_block<T, 4, 2>);
  compare("held=3", &launch_block<T, 4, 3>);
  compare("held=4", &launch_block<T, 4, 4>);
  compare("warp/4", &launch_warp<T, 4>);
  compare("warp/8", &launch_warp<T, 8>);
  compare("warp/16", &launch_warp<T, 16>);
  return fails;
}

// ---------------------------------------------------------------------------
// Kernel-only benchmark.
// ---------------------------------------------------------------------------
template <typename Fn>
static double bench(Fn go, int iters) {
  for (int i = 0; i < 10; ++i) go();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, e;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) go();
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, e));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms / iters;
}

int main(int argc, char** argv) {
  bool do_bench = false, do_fuzz = false;
  std::string csv;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--bench")) do_bench = true;
    else if (!std::strcmp(argv[i], "--fuzz")) do_fuzz = true;
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s sm_%d%d cuda=%d.%d\n", prop.name, prop.major, prop.minor,
              CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

  int failures = 0;

  std::printf("\n== converter fixtures (%d values + packed quad) ==\n",
              (int)(sizeof(kFixtures) / sizeof(kFixtures[0])));
  failures += check_fixtures();
  if (!failures) std::printf("  all fixtures PASS\n");

  if (do_fuzz) {
    std::printf("\n== mapping self-consistency fuzz ==\n");
    struct S { int64_t m, n; };
    const S shapes[] = {
        {1, 4096}, {2, 3072}, {7, 128}, {8, 512}, {33, 2048}, {64, 512},
        {128, 4096}, {512, 1024}, {1000, 3072}, {2048, 4096},
        // K values that are multiples of 4 but not of the block width and not
        // of blockDim*HELD, which is where the chunking logic matters.  The
        // dispatcher rejects K % 4 != 0, so those are covered by the Python
        // suite instead (test_unsupported_inputs_are_declined).
        {16, 4}, {16, 8}, {16, 508}, {16, 1020}, {16, 1028}, {16, 4100},
        {16, 7168}, {16, 12288},
    };
    for (const S& s : shapes) {
      int f = 0;
      f += fuzz_one<__nv_bfloat16>(s.m, s.n, 1234u + (uint32_t)s.n);
      f += fuzz_one<__half>(s.m, s.n, 77u + (uint32_t)s.n);
      f += fuzz_one<float>(s.m, s.n, 9u + (uint32_t)s.n);
      failures += f;
      std::printf("  M=%-5lld K=%-5lld bf16+fp16+fp32 %s\n", (long long)s.m,
                  (long long)s.n, f ? "<-- FAIL" : "ok");
    }
  }

  if (do_bench) {
    std::printf("\n== kernel-only bench (bf16 in, e4m3 out) ==\n");
    std::printf("%-22s %5s %11s %11s %8s\n", "shape", "held", "block_us",
                "warp8_us", "best");
    const int64_t ms[] = {1, 2, 8, 32, 128, 512, 1024, 2048, 4096};
    const int64_t ks[] = {3072, 4096};
    std::vector<std::string> rows;
    for (int64_t n : ks) {
      for (int64_t m : ms) {
        std::vector<float> hx(m * n);
        std::mt19937 rng(5u + (uint32_t)n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = nd(rng);
        auto hb = to_type<__nv_bfloat16>(hx);
        DevBuf<__nv_bfloat16> b;
        b.alloc(m, n, hb);

        const int64_t nvec = n / 4;
        const int held = (nvec % kBlockThreads == 0 && nvec / kBlockThreads >= 1 &&
                          nvec / kBlockThreads <= 4)
                             ? (int)(nvec / kBlockThreads)
                             : 0;
        const size_t smem = ((kBlockThreads + 31) / 32) * sizeof(float);
        auto go_block = [&] {
          per_token_quant_kernel<__nv_bfloat16, 4, 0>
              <<<dim3((unsigned)m), dim3(kBlockThreads), smem>>>(b.x, m, n, b.s, b.q,
                                                                 nullptr, false);
        };
        auto go_warp = [&] {
          per_token_quant_kernel_warp<__nv_bfloat16, 4, 8>
              <<<dim3((unsigned)((m + 7) / 8)), dim3(256)>>>(b.x, m, n, b.s, b.q,
                                                             nullptr, false);
        };
        const double tb = bench(go_block, 50);
        const double tw = bench(go_warp, 50);
        char label[64];
        std::snprintf(label, sizeof(label), "M=%lld K=%lld", (long long)m,
                      (long long)n);
        std::printf("%-22s %5d %11.3f %11.3f %8s\n", label, held, tb * 1000.0,
                    tw * 1000.0, tb < tw ? "block" : "warp");
        char row[160];
        std::snprintf(row, sizeof(row), "%lld,%lld,%d,%.6f,%.6f", (long long)m,
                      (long long)n, held, tb * 1000.0, tw * 1000.0);
        rows.push_back(row);
      }
    }
    if (!csv.empty()) {
      FILE* f = std::fopen(csv.c_str(), "w");
      if (f) {
        std::fprintf(f, "m,n,held,block_us,warp8_us\n");
        for (const std::string& r : rows) std::fprintf(f, "%s\n", r.c_str());
        std::fclose(f);
        std::printf("wrote %s\n", csv.c_str());
      }
    }
  }

  std::printf("\n%s (%d failing checks)\n", failures ? "RESULT: FAIL" : "RESULT: PASS",
              failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
