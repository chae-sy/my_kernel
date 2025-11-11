#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>         

__global__ void matmul_0_1(const half *A, const half *B, half *C,
                           int M, int N, int K) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N) {
        float acc = 0.f;
        for (int i = 0; i < K; i++) {
            __half a = A[row * K + i];   // row-major
            __half b = B[col * K + i];   // col-major (col index 먼저)
            float psum = __half2float(__hmul(a, b));
            acc += psum;
        }
        C[row * N + col] = __float2half(acc);
    }
}

// Kernel 1.0: Naive mma 
#include <mma.h>
#include <cuda_fp16.h>
#define M_TILE 2
#define N_TILE 2
__forceinline__ __device__
void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(D[0]),"=f"(D[1]),"=f"(D[2]),"=f"(D[3])
    : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
      "r"(B[0]),"r"(B[1]),
      "f"(C[0]),"f"(C[1]),"f"(C[2]),"f"(C[3])
  );
}

// blockDim.x == 32 (1 warp), grid = (N/8, M/16)
__global__ void mma_matmul_1_0(const half* __restrict__ A,
                               const half* __restrict__ B, // column-major
                               half* __restrict__ C,
                               int M, int N, int K) {
  const int lane = threadIdx.x;            // 0..31
  const int m0 = blockIdx.y * 16;          // 출력 블록 시작 행
  const int n0 = blockIdx.x * 8;           // 출력 블록 시작 열

  // 누산 레지스터
  float dReg[4] = {0.f, 0.f, 0.f, 0.f};

  // lane→(groupID, groupLane) 매핑
  const int groupID     = lane / 4;  // 0..7
  const int groupLaneID = lane % 4;  // 0..3

  // K를 16씩 루프
  for (int k0 = 0; k0 < K; k0 += 16) {
    // 각 mma 호출용 레지스터 조립 (A: 8 halves, B: 4 halves)
    half aReg[8];
    half bReg[4];

    // A: row-major, tile: [m0..m0+15]×[k0..k0+15]
    // (아래 인덱스 매핑은 NVIDIA warp 프래그먼트 매핑을 따름)
    int ar = m0 + (groupID);          // 0..15
    int ar8= m0 + (groupID + 8);      // 8..23(경계는 아래서 가드)
    int ac0= k0 + groupLaneID*2;      // 0,2,4,6
    int ac8= k0 + groupLaneID*2 + 8;  // 8,10,12,14

    auto ldA = [&](int r, int c)->half {
      if (r < M && c < K) return A[r * K + c];
      return __float2half(0.f);
    };
    aReg[0] = ldA(ar , ac0+0);
    aReg[1] = ldA(ar , ac0+1);
    aReg[2] = ldA(ar8, ac0+0);
    aReg[3] = ldA(ar8, ac0+1);
    aReg[4] = ldA(ar , ac8+0);
    aReg[5] = ldA(ar , ac8+1);
    aReg[6] = ldA(ar8, ac8+0);
    aReg[7] = ldA(ar8, ac8+1);

    // B: column-major, tile: [k0..k0+15]×[n0..n0+7]
    int br0 = k0 + groupLaneID*2; // 0,2,4,6
    int br8 = k0 + groupLaneID*2 + 8;
    int bc  = n0 + groupID;       // 0..7

    auto ldBcm = [&](int r, int c)->half {
      if (r < K && c < N) return B[c * K + r]; // column-major
      return __float2half(0.f);
    };
    bReg[0] = ldBcm(br0+0, bc);
    bReg[1] = ldBcm(br0+1, bc);
    bReg[2] = ldBcm(br8+0, bc);
    bReg[3] = ldBcm(br8+1, bc);

    // 레지스터 포인터로 MMA 진행
    const unsigned* aPtr = reinterpret_cast<const unsigned*>(&aReg[0]);
    const unsigned* bPtr = reinterpret_cast<const unsigned*>(&bReg[0]);
    mma_m16n8k16(aPtr, bPtr, dReg, dReg);
  }

  // 결과 저장 (half로 캐스팅, 경계 가드)
  auto stC = [&](int r, int c, float v) {
    if (r < M && c < N) C[r * N + c] = __float2half(v);
  };
  int r0 = m0 + groupID;
  int r8 = m0 + groupID + 8;
  int c0 = n0 + groupLaneID*2;

  stC(r0, c0+0, dReg[0]);
  stC(r0, c0+1, dReg[1]);
  stC(r8, c0+0, dReg[2]);
  stC(r8, c0+1, dReg[3]);
}


// Kernel 1.1: Naive mma with 2x M/N tiling
__launch_bounds__(16 * 16)
__global__ void mma_matmul_1_1(const half *A, const half *B, half *C, int M, int N, int K) {
  // declare cache in shared memory
  __shared__ half As[M_TILE * 32][16];
  __shared__ half Bs[16][N_TILE * 32];

  int mBlock = M_TILE * 32 * blockIdx.y;
  int nBlock = N_TILE * 32 * blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;

  // tile warps as follows
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int nWarp = 8 * (warpID % 4);
  int mWarp = 16 * (warpID / 4);

  int groupID     = laneID / 4;
  int groupLaneID = laneID % 4;

  half  aReg[8];
  half  bReg[4];
  float dReg[M_TILE][N_TILE][4] = {0.};

  for (int kStart=0; kStart < K; kStart += 16) {
    for (int m=0; m < M_TILE; ++m) {
      int mTile = m * 32;
      As[mTile      + ty][tx] = A[(mBlock + mTile      + ty)*K + kStart + tx];
      As[mTile + 16 + ty][tx] = A[(mBlock + mTile + 16 + ty)*K + kStart + tx];
    }
    for (int n=0; n < N_TILE; ++n) {
      int nTile = n * 32;
      // (수정, col-major: offset = (nBlock + nTile + tx)*K + (kStart + ty))
Bs[ty][nTile      + tx] = B[(nBlock + nTile      + tx) * K + (kStart + ty)];
Bs[ty][nTile + 16 + tx] = B[(nBlock + nTile + 16 + tx) * K + (kStart + ty)];
    }
    __syncthreads();
    for (int m=0; m < M_TILE; m++) {
      int mTile = m * 32;
      // set up the registers for mma call
      aReg[0] = As[mTile + mWarp + groupID    ][groupLaneID*2    ];
      aReg[1] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 1];
      aReg[2] = As[mTile + mWarp + groupID + 8][groupLaneID*2    ];
      aReg[3] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 1];
      aReg[4] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 8];
      aReg[5] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 9];
      aReg[6] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 8];
      aReg[7] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 9];
      for (int n=0; n < N_TILE; n++) {
        int nTile = n * 32;
        bReg[0] = Bs[groupLaneID*2 + 0][nTile + nWarp + groupID];
        bReg[1] = Bs[groupLaneID*2 + 1][nTile + nWarp + groupID];
        bReg[2] = Bs[groupLaneID*2 + 8][nTile + nWarp + groupID];
        bReg[3] = Bs[groupLaneID*2 + 9][nTile + nWarp + groupID];

        const unsigned *aPtr = reinterpret_cast<const unsigned*>(aReg);   // 또는 &aReg[0]
const unsigned *bPtr = reinterpret_cast<const unsigned*>(bReg);   // 또는 &bReg[0]

        mma_m16n8k16(aPtr, bPtr, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }
  // Copy dReg to global memory
  for (int m=0; m < M_TILE; m++) {
    int mTile = m * 32;
    for (int n=0; n < N_TILE; n++) {
      int nTile = n * 32;
      C[(mBlock + mTile + mWarp + groupID  )*N + nBlock + nTile + nWarp + 2*groupLaneID  ] = __float2half(dReg[m][n][0]);
      C[(mBlock + mTile + mWarp + groupID  )*N + nBlock + nTile + nWarp + 2*groupLaneID+1] = __float2half(dReg[m][n][1]);
      C[(mBlock + mTile + mWarp + groupID+8)*N + nBlock + nTile + nWarp + 2*groupLaneID  ] = __float2half(dReg[m][n][2]);
      C[(mBlock + mTile + mWarp + groupID+8)*N + nBlock + nTile + nWarp + 2*groupLaneID+1] = __float2half(dReg[m][n][3]);
    }
  }
}



