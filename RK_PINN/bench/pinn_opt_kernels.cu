// pinn_opt_kernels.cu — optimised PINN_V2_UTT forward-pass variants (NVRTC TU).
//
// Architecture (locked): 6 -> 96 -> 96 -> 4, tanh on the two hidden layers,
// linear head, fp32. Weights are the constexpr arrays baked into the locked
// header PINN_V2_UTT.cuh (kW0[576], kW1[9216], kW2[384] + biases).
//
// Reference path  : pinn_ref  (calls the locked pinn_v2_utt_state verbatim).
// Optimised paths : pinn_fused, pinn_warp, pinn_warp_u4 (see below).
//
// Envelope (spec §3), identical across all variants:
//   x'  = x + tx*dz + c2*dz ;  y' = y + ty*dz + c3*dz ;
//   tx' = tx + c0          ;  ty' = ty + c1           ; qop unchanged.

#include "ParKalmanDefinitions.cuh"   // KalmanFloat = float
#include "PINN_V2_UTT.cuh"            // locked weights + reference pinn_v2_utt_state

namespace W = ParKalmanFilter::PINN_V2_UTT_Weights;

// ---- shared envelope helper -------------------------------------------------
__device__ __forceinline__ void envelope(
    float x, float y, float tx, float ty, float dz,
    float c0, float c1, float c2, float c3,
    float& ox, float& oy, float& otx, float& oty)
{
  float xo = x; xo = fmaf(tx, dz, xo); xo = fmaf(c2, dz, xo);
  float yo = y; yo = fmaf(ty, dz, yo); yo = fmaf(c3, dz, yo);
  ox = xo; oy = yo; otx = tx + c0; oty = ty + c1;
}

// =============================================================================
// 0. REFERENCE — verbatim locked kernel (same-slot baseline for accuracy/timing)
// =============================================================================
extern "C" __global__ void pinn_ref(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float xo, yo, txo, tyo;
  ParKalmanFilter::pinn_v2_utt_state(X[i], Y[i], TX[i], TY[i], QOP[i], DZ[i], xo, yo, txo, tyo);
  OX[i] = xo; OY[i] = yo; OTX[i] = txo; OTY[i] = tyo;
}

// =============================================================================
// 1. FUSED — thread-per-track, h0 register-resident, head fused into L1 loop
//    so h1 never materialises (no local-memory spill). Weights stay constexpr
//    => warp-uniform constant-memory broadcast (the property that already works).
// =============================================================================
extern "C" __global__ void pinn_fused(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  using namespace W;

  const float x = X[t], y = Y[t], tx = TX[t], ty = TY[t], qop = QOP[t], dz = DZ[t];
  const float in6[6] = {
    (x   - kInputMean[0]) * (1.0f / kInputStd[0]),
    (y   - kInputMean[1]) * (1.0f / kInputStd[1]),
    (tx  - kInputMean[2]) * (1.0f / kInputStd[2]),
    (ty  - kInputMean[3]) * (1.0f / kInputStd[3]),
    (qop - kInputMean[4]) * (1.0f / kInputStd[4]),
    1.0f };

  // layer 0: 6 -> 96, h0 held in registers (indices constant after unroll).
  float h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }

  // layer 1 (96->96, tanh) FUSED with head (96->4, linear): each h1[j] is
  // consumed immediately into the 4 accumulators, never stored -> no h1 spill.
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0);
    c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2);
    c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }

  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}

// =============================================================================
//  Weight-init kernel: transpose the locked constexpr weights into global
//  buffers (input-major) so the warp-cooperative kernels read them coalesced.
//    Wt0[i*96+o] = kW0[o*6+i]   (6x96)
//    Wt1[i*96+o] = kW1[o*96+i]  (96x96)
//    W2 row-major kW2 copied as-is (4x96); biases copied as-is.
// =============================================================================
extern "C" __global__ void init_weights(
  float* __restrict__ Wt0, float* __restrict__ Wt1, float* __restrict__ W2c,
  float* __restrict__ B0, float* __restrict__ B1, float* __restrict__ B2)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  using namespace W;
  // Wt1 (96*96) is the big one — one thread per element.
  for (int e = idx; e < 96 * 96; e += gridDim.x * blockDim.x) {
    const int o = e / 96, i = e % 96;     // original [o][i]
    Wt1[i * 96 + o] = kW1[o * 96 + i];    // transposed [i][o]
  }
  for (int e = idx; e < 6 * 96; e += gridDim.x * blockDim.x) {
    const int o = e / 6, i = e % 6;       // kW0 is [96][6]
    Wt0[i * 96 + o] = kW0[o * 6 + i];
  }
  for (int e = idx; e < 4 * 96; e += gridDim.x * blockDim.x) W2c[e] = kW2[e];
  for (int e = idx; e < 96; e += gridDim.x * blockDim.x) { B0[e] = kB0[e]; B1[e] = kB1[e]; }
  for (int e = idx; e < 4;  e += gridDim.x * blockDim.x) B2[e] = kB2[e];
}

// =============================================================================
// 2. WARP-COOPERATIVE GEMV — one track per warp, 96 neurons across 32 lanes
//    (3/lane). Activations live in shared (broadcast reads); weights come from
//    transposed global buffers (coalesced lane reads). Slashes the 192-deep
//    serial-tanh latency chain of the single-thread path to ~6 tanh/lane.
// =============================================================================
__device__ __forceinline__ void warp_body(
  int gw, int lane, int wib, int warps_per_block, float* smem,
  const float* X, const float* Y, const float* TX, const float* TY,
  const float* QOP, const float* DZ,
  const float* Wt0, const float* Wt1, const float* W2c,
  const float* B0, const float* B1, const float* B2,
  float* OX, float* OY, float* OTX, float* OTY, int unroll4)
{
  using namespace W;
  float* h0 = smem + wib * 96;
  float* h1 = smem + warps_per_block * 96 + wib * 96;

  const float x = X[gw], y = Y[gw], tx = TX[gw], ty = TY[gw], qop = QOP[gw], dz = DZ[gw];
  const float in6[6] = {
    (x   - kInputMean[0]) * (1.0f / kInputStd[0]),
    (y   - kInputMean[1]) * (1.0f / kInputStd[1]),
    (tx  - kInputMean[2]) * (1.0f / kInputStd[2]),
    (ty  - kInputMean[3]) * (1.0f / kInputStd[3]),
    (qop - kInputMean[4]) * (1.0f / kInputStd[4]),
    1.0f };

  // layer 0: 6 -> 96, lane owns outputs {lane, lane+32, lane+64}
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int o = lane + 32 * k;
    float a = B0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(Wt0[i * 96 + o], in6[i], a);
    h0[o] = tanhf(a);
  }
  __syncwarp();

  // layer 1: 96 -> 96, coalesced Wt1 lane reads, h0 broadcast from shared
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int o = lane + 32 * k;
    float a = B1[o];
    if (unroll4) {
      #pragma unroll 4
      for (int i = 0; i < 96; ++i) a = fmaf(Wt1[i * 96 + o], h0[i], a);
    } else {
      for (int i = 0; i < 96; ++i) a = fmaf(Wt1[i * 96 + o], h0[i], a);
    }
    h1[o] = tanhf(a);
  }
  __syncwarp();

  // head: 96 -> 4, lanes 0..3 each compute one c
  float myc = 0.f;
  if (lane < 4) {
    float a = B2[lane];
    for (int i = 0; i < 96; ++i) a = fmaf(W2c[lane * 96 + i], h1[i], a);
    myc = a;
  }
  const float c0 = __shfl_sync(0xffffffffu, myc, 0);
  const float c1 = __shfl_sync(0xffffffffu, myc, 1);
  const float c2 = __shfl_sync(0xffffffffu, myc, 2);
  const float c3 = __shfl_sync(0xffffffffu, myc, 3);

  if (lane == 0) {
    float ox, oy, otx, oty;
    envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
    OX[gw] = ox; OY[gw] = oy; OTX[gw] = otx; OTY[gw] = oty;
  }
}

extern "C" __global__ void pinn_warp(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, const float* __restrict__ Wt0, const float* __restrict__ Wt1,
  const float* __restrict__ W2c, const float* __restrict__ B0, const float* __restrict__ B1,
  const float* __restrict__ B2, float* __restrict__ OX, float* __restrict__ OY,
  float* __restrict__ OTX, float* __restrict__ OTY)
{
  extern __shared__ float smem[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gw = tid >> 5, lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int wpb = blockDim.x >> 5;
  if (gw >= N) return;
  warp_body(gw, lane, wib, wpb, smem, X, Y, TX, TY, QOP, DZ,
            Wt0, Wt1, W2c, B0, B1, B2, OX, OY, OTX, OTY, 0);
}

extern "C" __global__ void pinn_warp_u4(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, const float* __restrict__ Wt0, const float* __restrict__ Wt1,
  const float* __restrict__ W2c, const float* __restrict__ B0, const float* __restrict__ B1,
  const float* __restrict__ B2, float* __restrict__ OX, float* __restrict__ OY,
  float* __restrict__ OTX, float* __restrict__ OTY)
{
  extern __shared__ float smem[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int gw = tid >> 5, lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int wpb = blockDim.x >> 5;
  if (gw >= N) return;
  warp_body(gw, lane, wib, wpb, smem, X, Y, TX, TY, QOP, DZ,
            Wt0, Wt1, W2c, B0, B1, B2, OX, OY, OTX, OTY, 1);
}
