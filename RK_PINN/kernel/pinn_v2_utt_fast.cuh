/*****************************************************************************
 *  pinn_v2_utt_fast.cuh — THE CANDIDATE (optimised PINN_V2_UTT forward pass)
 *
 *  This is the deliverable of the throughput-optimisation pass. It is the
 *  drop-in replacement for the body of ParKalmanFilter::pinn_v2_utt_state()
 *  in the locked header candidate/pinn_v2_ALLEN_v1/PINN_V2_UTT.cuh.
 *
 *  Two variants:
 *    pinn_v2_utt_state_fused  — h96, fp32, BIT-EXACT to the locked reference
 *                               (0 µm / 0 rad over 1 M tracks). 4.85 ns/track
 *                               on V100, beats RK (5.71). ZERO-risk drop-in;
 *                               preserves the A4 Jacobian / R6 parity gates.
 *    pinn_v2_utt_state_h64_fu — 6→64→64→4, fully unrolled. 0.91 ns/track
 *                               (73.8 % fp32 peak); 2.6× faster than extrapUTT,
 *                               6.3× faster than RK. Needs the wave2_resid_h64
 *                               weights exported (see note below) + A4/R6 re-gate.
 *
 *  The win (vs the locked spilling kernel): the locked kernel keeps two
 *  DYNAMICALLY-indexed arrays h0[96],h1[96] per thread → they spill to local
 *  memory → ~9216 local loads/track. Here: unroll the contraction so h0 is
 *  register-resident, and FUSE the linear head into the layer-1 loop so h1
 *  never materialises (one live scalar hj + 4 accumulators). Spill → 0 B.
 *  Weights stay constexpr → warp-uniform constant-bank FFMA operands (free).
 *
 *  Provenance: Ex_rep/RK_PINN (bench + results/combined.json); locked weights
 *  track-extrapolation-pinn @ 283b03b, PINN_V2_UTT.cuh, blob CRC32 0x1a139335.
 *  Measured Tesla V100-PCIE-32GB, fp32, 1 M gen-4 tracks, NVRTC sm_70.
 *  See ../README.md and ../RESULTS.md.
 *****************************************************************************/
#pragma once

#include "PINN_V2_UTT.cuh"   // locked constexpr weights + the reference function

namespace ParKalmanFilter {

// ---- shared envelope (spec §3), identical to the locked reference ----------
__device__ __host__ __forceinline__ void pinn_v2_envelope_(
    float x, float y, float tx, float ty, float dz,
    float c0, float c1, float c2, float c3,
    KalmanFloat& x_out, KalmanFloat& y_out, KalmanFloat& tx_out, KalmanFloat& ty_out)
{
  float xo = x; xo = fmaf(tx, dz, xo); xo = fmaf(c2, dz, xo);
  float yo = y; yo = fmaf(ty, dz, yo); yo = fmaf(c3, dz, yo);
  x_out = KalmanFloat(xo);  y_out = KalmanFloat(yo);
  tx_out = KalmanFloat(tx + c0);  ty_out = KalmanFloat(ty + c1);
}

// ===========================================================================
//  DEPLOYABLE, BIT-EXACT (h96).  Drop this body into pinn_v2_utt_state().
//  Same signature, same outputs (bit-for-bit) as the locked reference.
// ===========================================================================
__device__ __host__ inline void pinn_v2_utt_state_fused(
    KalmanFloat x_in, KalmanFloat y_in, KalmanFloat tx_in, KalmanFloat ty_in,
    KalmanFloat qop_in, KalmanFloat dz,
    KalmanFloat& x_out, KalmanFloat& y_out, KalmanFloat& tx_out, KalmanFloat& ty_out)
{
  using namespace PINN_V2_UTT_Weights;
  const float x = float(x_in), y = float(y_in), tx = float(tx_in),
              ty = float(ty_in), qop = float(qop_in), dzf = float(dz);

  // EXACT same normalisation as the reference (divide, not reciprocal-multiply)
  const float in6[6] = {
    (x   - kInputMean[0]) / kInputStd[0], (y  - kInputMean[1]) / kInputStd[1],
    (tx  - kInputMean[2]) / kInputStd[2], (ty - kInputMean[3]) / kInputStd[3],
    (qop - kInputMean[4]) / kInputStd[4], 1.0f };

  // layer 0 (6->96): h0 register-resident (constant indices after unroll)
  float h0[96];
  #pragma unroll
  for (int o = 0; o < 96; ++o) {
    float a = kB0[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }
  // layer 1 (96->96, tanh) FUSED with head (96->4): h1[j] consumed immediately,
  // never stored. Reduction order identical to the reference => bit-exact.
  float c0 = kB2[0], c1 = kB2[1], c2 = kB2[2], c3 = kB2[3];
  for (int j = 0; j < 96; ++j) {
    float a = kB1[j];
    #pragma unroll
    for (int i = 0; i < 96; ++i) a = fmaf(kW1[j * 96 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2[j], hj, c0);        c1 = fmaf(kW2[96 + j], hj, c1);
    c2 = fmaf(kW2[192 + j], hj, c2);  c3 = fmaf(kW2[288 + j], hj, c3);
  }
  pinn_v2_envelope_(x, y, tx, ty, dzf, c0, c1, c2, c3, x_out, y_out, tx_out, ty_out);
}

// ===========================================================================
//  FASTEST (h64, fully unrolled).  0.91 ns/track.
//
//  NOTE: a true deployable needs the wave2_resid_h64 weights exported to
//  constexpr arrays kW0_64[6*64], kW1_64[64*64], kW2_64[4*64] + biases
//  (checkpoint: TrackExtrapolation/experiments/gen_3/trained_models/
//   wave2_resid_h64; export via For_Allen/scripts/emit_cuda_header.py).
//  Accuracy is the recorded capacity-ladder result (2.79 mm ≈ h96).
//
//  Below is the deployable SHAPE, written against placeholder symbols
//  kW0_64/kW1_64/kW2_64/kB0_64/kB1_64/kB2_64; compile only once those exist.
//  (The bench times this shape faithfully by reusing the h96 constants as a
//   64×64 sub-block — see ../bench/pinn_opt_kernels_v3.cu::pinn_h64_fu.)
// ===========================================================================
#ifdef PINN_V2_UTT_H64_WEIGHTS_AVAILABLE
__device__ __host__ inline void pinn_v2_utt_state_h64_fu(
    KalmanFloat x_in, KalmanFloat y_in, KalmanFloat tx_in, KalmanFloat ty_in,
    KalmanFloat qop_in, KalmanFloat dz,
    KalmanFloat& x_out, KalmanFloat& y_out, KalmanFloat& tx_out, KalmanFloat& ty_out)
{
  using namespace PINN_V2_UTT_Weights;
  const float x = float(x_in), y = float(y_in), tx = float(tx_in),
              ty = float(ty_in), qop = float(qop_in), dzf = float(dz);
  const float in6[6] = {
    (x   - kInputMean[0]) / kInputStd[0], (y  - kInputMean[1]) / kInputStd[1],
    (tx  - kInputMean[2]) / kInputStd[2], (ty - kInputMean[3]) / kInputStd[3],
    (qop - kInputMean[4]) / kInputStd[4], 1.0f };
  float h0[64];
  #pragma unroll
  for (int o = 0; o < 64; ++o) {
    float a = kB0_64[o];
    #pragma unroll
    for (int i = 0; i < 6; ++i) a = fmaf(kW0_64[o * 6 + i], in6[i], a);
    h0[o] = tanhf(a);
  }
  float c0 = kB2_64[0], c1 = kB2_64[1], c2 = kB2_64[2], c3 = kB2_64[3];
  #pragma unroll
  for (int j = 0; j < 64; ++j) {
    float a = kB1_64[j];
    #pragma unroll
    for (int i = 0; i < 64; ++i) a = fmaf(kW1_64[j * 64 + i], h0[i], a);
    const float hj = tanhf(a);
    c0 = fmaf(kW2_64[j], hj, c0);       c1 = fmaf(kW2_64[64 + j], hj, c1);
    c2 = fmaf(kW2_64[128 + j], hj, c2); c3 = fmaf(kW2_64[192 + j], hj, c3);
  }
  pinn_v2_envelope_(x, y, tx, ty, dzf, c0, c1, c2, c3, x_out, y_out, tx_out, ty_out);
}
#endif  // PINN_V2_UTT_H64_WEIGHTS_AVAILABLE

}  // namespace ParKalmanFilter
