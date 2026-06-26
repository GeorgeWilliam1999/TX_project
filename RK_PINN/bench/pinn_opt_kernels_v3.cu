// pinn_opt_kernels_v3.cu — exhaust the remaining levers on the throughput winner.
//
//   pinn_ref        : verbatim locked (parity ref + same-slot baseline)
//   pinn_fused      : current best (inner-unrolled, j rolled, bit-exact)
//   pinn_fused_fu   : FULL unroll of both layer-1 loops -> weights become
//                     immediate constant-bank FFMA operands (no LDC). bit-exact.
//   pinn_fused_ilp4 : 4-way accumulator split of the inner reduction (breaks the
//                     96-deep dependency chain -> ILP). NOT bit-exact (reorders).
//   pinn_h64        : 6->64->64->4 fused (accuracy-equivalent width), inner-unroll
//   pinn_h64_fu     : h64 fully unrolled

#include "ParKalmanDefinitions.cuh"
#include "PINN_V2_UTT.cuh"
namespace W = ParKalmanFilter::PINN_V2_UTT_Weights;

__device__ __forceinline__ void envelope(
    float x, float y, float tx, float ty, float dz,
    float c0, float c1, float c2, float c3,
    float& ox, float& oy, float& otx, float& oty) {
  float xo = x; xo = fmaf(tx, dz, xo); xo = fmaf(c2, dz, xo);
  float yo = y; yo = fmaf(ty, dz, yo); yo = fmaf(c3, dz, yo);
  ox = xo; oy = yo; otx = tx + c0; oty = ty + c1;
}
#define NORM6(x,y,tx,ty,qop) { \
    (x-kInputMean[0])/kInputStd[0], (y-kInputMean[1])/kInputStd[1], \
    (tx-kInputMean[2])/kInputStd[2], (ty-kInputMean[3])/kInputStd[3], \
    (qop-kInputMean[4])/kInputStd[4], 1.0f }
#define TID const int t = blockIdx.x*blockDim.x+threadIdx.x; if (t>=N) return;
#define LOADIN using namespace W; \
    const float x=X[t],y=Y[t],tx=TX[t],ty=TY[t],qop=QOP[t],dz=DZ[t]; \
    const float in6[6]=NORM6(x,y,tx,ty,qop);
#define STORE float ox,oy,otx,oty; envelope(x,y,tx,ty,dz,c0,c1,c2,c3,ox,oy,otx,oty); \
    OX[t]=ox; OY[t]=oy; OTX[t]=otx; OTY[t]=oty;
#define SIG const float* __restrict__ X,const float* __restrict__ Y,const float* __restrict__ TX, \
    const float* __restrict__ TY,const float* __restrict__ QOP,const float* __restrict__ DZ, \
    const int N,float* __restrict__ OX,float* __restrict__ OY,float* __restrict__ OTX,float* __restrict__ OTY

extern "C" __global__ void pinn_ref(SIG) {
  TID; float xo,yo,txo,tyo;
  ParKalmanFilter::pinn_v2_utt_state(X[t],Y[t],TX[t],TY[t],QOP[t],DZ[t],xo,yo,txo,tyo);
  OX[t]=xo;OY[t]=yo;OTX[t]=txo;OTY[t]=tyo;
}

extern "C" __global__ void pinn_fused(SIG) {
  TID; LOADIN;
  float h0[96];
  #pragma unroll
  for (int o=0;o<96;++o){ float a=kB0[o];
    #pragma unroll
    for (int i=0;i<6;++i) a=fmaf(kW0[o*6+i],in6[i],a); h0[o]=tanhf(a); }
  float c0=kB2[0],c1=kB2[1],c2=kB2[2],c3=kB2[3];
  for (int j=0;j<96;++j){ float a=kB1[j];
    #pragma unroll
    for (int i=0;i<96;++i) a=fmaf(kW1[j*96+i],h0[i],a);
    float hj=tanhf(a);
    c0=fmaf(kW2[j],hj,c0); c1=fmaf(kW2[96+j],hj,c1);
    c2=fmaf(kW2[192+j],hj,c2); c3=fmaf(kW2[288+j],hj,c3); }
  STORE;
}

extern "C" __global__ void pinn_fused_fu(SIG) {
  TID; LOADIN;
  float h0[96];
  #pragma unroll
  for (int o=0;o<96;++o){ float a=kB0[o];
    #pragma unroll
    for (int i=0;i<6;++i) a=fmaf(kW0[o*6+i],in6[i],a); h0[o]=tanhf(a); }
  float c0=kB2[0],c1=kB2[1],c2=kB2[2],c3=kB2[3];
  #pragma unroll
  for (int j=0;j<96;++j){ float a=kB1[j];
    #pragma unroll
    for (int i=0;i<96;++i) a=fmaf(kW1[j*96+i],h0[i],a);
    float hj=tanhf(a);
    c0=fmaf(kW2[j],hj,c0); c1=fmaf(kW2[96+j],hj,c1);
    c2=fmaf(kW2[192+j],hj,c2); c3=fmaf(kW2[288+j],hj,c3); }
  STORE;
}

extern "C" __global__ void pinn_fused_ilp4(SIG) {
  TID; LOADIN;
  float h0[96];
  #pragma unroll
  for (int o=0;o<96;++o){ float a=kB0[o];
    #pragma unroll
    for (int i=0;i<6;++i) a=fmaf(kW0[o*6+i],in6[i],a); h0[o]=tanhf(a); }
  float c0=kB2[0],c1=kB2[1],c2=kB2[2],c3=kB2[3];
  for (int j=0;j<96;++j){
    float a0=kB1[j],a1=0.f,a2=0.f,a3=0.f;
    #pragma unroll
    for (int i=0;i<96;i+=4){
      a0=fmaf(kW1[j*96+i],  h0[i],  a0); a1=fmaf(kW1[j*96+i+1],h0[i+1],a1);
      a2=fmaf(kW1[j*96+i+2],h0[i+2],a2); a3=fmaf(kW1[j*96+i+3],h0[i+3],a3); }
    float hj=tanhf((a0+a1)+(a2+a3));
    c0=fmaf(kW2[j],hj,c0); c1=fmaf(kW2[96+j],hj,c1);
    c2=fmaf(kW2[192+j],hj,c2); c3=fmaf(kW2[288+j],hj,c3); }
  STORE;
}

extern "C" __global__ void pinn_h64(SIG) {
  TID; LOADIN;
  float h0[64];
  #pragma unroll
  for (int o=0;o<64;++o){ float a=kB0[o];
    #pragma unroll
    for (int i=0;i<6;++i) a=fmaf(kW0[o*6+i],in6[i],a); h0[o]=tanhf(a); }
  float c0=kB2[0],c1=kB2[1],c2=kB2[2],c3=kB2[3];
  for (int j=0;j<64;++j){ float a=kB1[j];
    #pragma unroll
    for (int i=0;i<64;++i) a=fmaf(kW1[j*96+i],h0[i],a);
    float hj=tanhf(a);
    c0=fmaf(kW2[j],hj,c0); c1=fmaf(kW2[96+j],hj,c1);
    c2=fmaf(kW2[192+j],hj,c2); c3=fmaf(kW2[288+j],hj,c3); }
  STORE;
}

extern "C" __global__ void pinn_h64_fu(SIG) {
  TID; LOADIN;
  float h0[64];
  #pragma unroll
  for (int o=0;o<64;++o){ float a=kB0[o];
    #pragma unroll
    for (int i=0;i<6;++i) a=fmaf(kW0[o*6+i],in6[i],a); h0[o]=tanhf(a); }
  float c0=kB2[0],c1=kB2[1],c2=kB2[2],c3=kB2[3];
  #pragma unroll
  for (int j=0;j<64;++j){ float a=kB1[j];
    #pragma unroll
    for (int i=0;i<64;++i) a=fmaf(kW1[j*96+i],h0[i],a);
    float hj=tanhf(a);
    c0=fmaf(kW2[j],hj,c0); c1=fmaf(kW2[96+j],hj,c1);
    c2=fmaf(kW2[192+j],hj,c2); c3=fmaf(kW2[288+j],hj,c3); }
  STORE;
}
