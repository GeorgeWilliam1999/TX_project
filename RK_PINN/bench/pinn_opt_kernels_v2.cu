// pinn_opt_kernels_v2.cu — round-2 optimised PINN_V2_UTT variants (NVRTC TU).
//
//   pinn_ref         : verbatim locked kernel (parity ref + same-slot baseline)
//   pinn_fused       : thread/track, register-resident h0, head fused into L1,
//                      EXACT divide normalisation => bit-parity with reference.
//   pinn_fused_ftanh : as pinn_fused but a fast __expf-based tanh (reduced-acc arm)
//   pinn_fused_h16   : h0 stored as __half (halved reg pressure -> occupancy);
//                      compute in fp32 (reduced-precision arm)
//   pinn_fused_lb    : pinn_fused with __launch_bounds__(256,3) (occupancy probe)
//   pinn_h64         : 6->64->64->4-shaped fused kernel (constant-broadcast
//                      weights, faithful timing of the accuracy-equivalent h64)
//
// Envelope + reductions match the locked reference exactly where parity is claimed.

#include "ParKalmanDefinitions.cuh"
#include "PINN_V2_UTT.cuh"
#include <cuda_fp16.h>

namespace W = ParKalmanFilter::PINN_V2_UTT_Weights;

__device__ __forceinline__ void envelope(
    float x, float y, float tx, float ty, float dz,
    float c0, float c1, float c2, float c3,
    float& ox, float& oy, float& otx, float& oty)
{
  float xo = x; xo = fmaf(tx, dz, xo); xo = fmaf(c2, dz, xo);
  float yo = y; yo = fmaf(ty, dz, yo); yo = fmaf(c3, dz, yo);
  ox = xo; oy = yo; otx = tx + c0; oty = ty + c1;
}

// fast tanh: tanh(x) = sign(x) * (1-e)/(1+e), e = exp(-2|x|)  (one SFU exp)
__device__ __forceinline__ float ftanh(float x) {
  float e = __expf(-2.0f * fabsf(x));
  float r = (1.0f - e) / (1.0f + e);
  return copysignf(r, x);
}

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

// ---- shared body macro for the thread-per-track fused variants ----
#define NORM6(x,y,tx,ty,qop) { \
    (x   - kInputMean[0]) / kInputStd[0], \
    (y   - kInputMean[1]) / kInputStd[1], \
    (tx  - kInputMean[2]) / kInputStd[2], \
    (ty  - kInputMean[3]) / kInputStd[3], \
    (qop - kInputMean[4]) / kInputStd[4], 1.0f }

// bit-parity fused kernel
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
  const float in6[6] = NORM6(x,y,tx,ty,qop);
  float h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0); c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2); c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }
  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}

// fast-tanh arm
extern "C" __global__ void pinn_fused_ftanh(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  using namespace W;
  const float x = X[t], y = Y[t], tx = TX[t], ty = TY[t], qop = QOP[t], dz = DZ[t];
  const float in6[6] = NORM6(x,y,tx,ty,qop);
  float h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = ftanh(a);
  }
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = ftanh(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0); c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2); c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }
  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}

// fp16-storage arm: h0 kept as __half (halves register footprint), fp32 compute
extern "C" __global__ void pinn_fused_h16(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  using namespace W;
  const float x = X[t], y = Y[t], tx = TX[t], ty = TY[t], qop = QOP[t], dz = DZ[t];
  const float in6[6] = NORM6(x,y,tx,ty,qop);
  __half h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = __float2half(tanhf(a));
  }
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], __half2float(h0[i]), a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0); c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2); c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }
  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}

// launch-bounds occupancy probe (force compiler toward fewer regs)
extern "C" __global__ void __launch_bounds__(256, 3) pinn_fused_lb(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  using namespace W;
  const float x = X[t], y = Y[t], tx = TX[t], ty = TY[t], qop = QOP[t], dz = DZ[t];
  const float in6[6] = NORM6(x,y,tx,ty,qop);
  float h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0); c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2); c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }
  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}

// h64-shaped fused kernel: faithful timing of the accuracy-equivalent 6->64->64->4
// net using constant-broadcast weights (reuses the locked constant arrays as a
// 64x64 sub-block; values irrelevant for timing, MAC/footprint pattern is exact).
extern "C" __global__ void pinn_h64(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  using namespace W;
  const float x = X[t], y = Y[t], tx = TX[t], ty = TY[t], qop = QOP[t], dz = DZ[t];
  const float in6[6] = NORM6(x,y,tx,ty,qop);
  float h0[64];
  #pragma unroll
  for (int o = 0; o < 64; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 64; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 64; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[0 * 96 + j], hj, c0); c1 = fmaf(kW2[1 * 96 + j], hj, c1);
    c2 = fmaf(kW2[2 * 96 + j], hj, c2); c3 = fmaf(kW2[3 * 96 + j], hj, c3);
  }
  float ox, oy, otx, oty;
  envelope(x, y, tx, ty, dz, c0, c1, c2, c3, ox, oy, otx, oty);
  OX[t] = ox; OY[t] = oy; OTX[t] = otx; OTY[t] = oty;
}
