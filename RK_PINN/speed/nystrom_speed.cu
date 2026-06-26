// nystrom_speed.cu — GPU timing of Allen's FAST extrapolator (RungeKuttaNystrom
// make_fast_step) vs the Cash-Karp RK, over the same gen-4 population & field
// texture as the throughput bench. The Nystrom struct can't be NVRTC-compiled
// verbatim (C++20 requires/convertible_to — see bench_rk.cuh note), so make_fast_step
// is reproduced BYTE-FAITHFULLY here (gamma + fillStages nodes {0,0.5,0.5,1}, one
// midpoint field eval/step) from RungeKuttaExtrapolator.cuh:63-145. State, the field
// texture path and the Cash-Karp RK are the verbatim Allen device code.

#include "bench_rk.cuh"   // verbatim: Magfield (3x tex3D), State, RungeKuttaExtrapolator<CashKarp>

using Extrapolators::State;

// ---- verbatim Cash-Karp path (identical to bench_kernels.cu::rk_kernel) ----
extern "C" __global__ void rk_kernel(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ Z0,
  const float* __restrict__ DZ, const int N, const float step_dz, const int max_steps,
  const cudaTextureObject_t tex_Bx, const cudaTextureObject_t tex_By, const cudaTextureObject_t tex_Bz,
  const float minX, const float minY, const float minZ, const float invDx, const float invDy,
  const float invDz, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  MagneticField::Magfield field;
  field.tex_Bx = tex_Bx; field.tex_By = tex_By; field.tex_Bz = tex_Bz;
  field.minX = minX; field.minY = minY; field.minZ = minZ;
  field.invDx = invDx; field.invDy = invDy; field.invDz = invDz;
  State s {X[i], Y[i], Z0[i], TX[i], TY[i], QOP[i]};
  const float target = Z0[i] + DZ[i];
  const float dir = (DZ[i] >= 0.f) ? 1.f : -1.f;
  for (int step = 0; step < max_steps; step++) {
    const float remaining = target - s.z;
    if (fabsf(remaining) < 0.5f) break;
    const float h = dir * fminf(step_dz, fabsf(remaining));
    State::Error err;
    Extrapolators::RungeKuttaExtrapolator<float, ButcherTableau::CashKarp<float>>::propagate(s, err, h, field);
  }
  OX[i] = s.x; OY[i] = s.y; OTX[i] = s.tx; OTY[i] = s.ty;
}

// ---- Allen RungeKuttaNystrom::make_fast_step (faithful), 1 field eval / step ----
__device__ __forceinline__ void nys_gamma(float tx, float ty, float Bx, float By, float Bz,
                                           float& gx, float& gy) {
  const float n = sqrtf(1.f + tx * tx + ty * ty);
  gx = n * (tx * ty * Bx - (1.f + tx * tx) * By + ty * Bz);
  gy = n * ((1.f + ty * ty) * Bx - tx * ty * By - tx * Bz);
}

extern "C" __global__ void nystrom_kernel(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ Z0,
  const float* __restrict__ DZ, const int N, const float step_dz, const int max_steps,
  const cudaTextureObject_t tex_Bx, const cudaTextureObject_t tex_By, const cudaTextureObject_t tex_Bz,
  const float minX, const float minY, const float minZ, const float invDx, const float invDy,
  const float invDz, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  MagneticField::Magfield field;
  field.tex_Bx = tex_Bx; field.tex_By = tex_By; field.tex_Bz = tex_Bz;
  field.minX = minX; field.minY = minY; field.minZ = minZ;
  field.invDx = invDx; field.invDy = invDy; field.invDz = invDz;

  float x = X[i], y = Y[i], z = Z0[i], tx = TX[i], ty = TY[i];
  const float qop = QOP[i];
  const float target = Z0[i] + DZ[i];
  const float dir = (DZ[i] >= 0.f) ? 1.f : -1.f;
  const float c[4] = {0.f, 0.5f, 0.5f, 1.f};

  for (int step = 0; step < max_steps; step++) {
    const float remaining = target - z;
    if (fabsf(remaining) < 0.5f) break;
    const float h = dir * fminf(step_dz, fabsf(remaining));
    const float3 B = field.fieldVectorLinearInterpolation(
        make_float3(x + 0.5f * tx * h, y + 0.5f * ty * h, z + 0.5f * h));
    float kx[4], ky[4], gx, gy;
    nys_gamma(tx, ty, B.x, B.y, B.z, gx, gy); kx[0] = qop * gx; ky[0] = qop * gy;
    #pragma unroll
    for (int st = 1; st < 4; st++) {
      const float tnx = tx + kx[st - 1] * (h * c[st]);
      const float tny = ty + ky[st - 1] * (h * c[st]);
      nys_gamma(tnx, tny, B.x, B.y, B.z, gx, gy); kx[st] = qop * gx; ky[st] = qop * gy;
    }
    const float h2 = h * h * (1.f / 6.f), h6 = h * (1.f / 6.f);
    x += tx * h + (kx[0] + kx[1] + kx[2]) * h2;
    y += ty * h + (ky[0] + ky[1] + ky[2]) * h2;
    tx += (kx[0] + 2.f * kx[1] + 2.f * kx[2] + kx[3]) * h6;
    ty += (ky[0] + 2.f * ky[1] + 2.f * ky[2] + ky[3]) * h6;
    z += h;
  }
  OX[i] = x; OY[i] = y; OTX[i] = tx; OTY[i] = ty;
}
