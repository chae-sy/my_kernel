#include <cuda_fp16.h>
#include <cstdio>
#define TILE_M 16
#define TILE_N 8
#define TILE_K 16

// --- ceil_div 매크로 (CUDA용) ---
#define ceil_div(x, y) (((x) + (y) - 1) / (y))


// --- 1. MMA intrinsic 정의 -----------------------------------------------
__device__ void mma_m16n8k16_f16( // C16×8​ = A16×16 ​× B16×8​ + C16×8​
    // Warp(32 threads) 가 협력해서 한 타일을 계산하고,
    // 각 스레드가 A, B, C, D 행렬의 “조각(fragment)”만을 맡아.
    float D[4], const half A[8], const half B[4], const float C[4]) {

    // half → 16bit pattern reinterpret
    const unsigned short *A_s = reinterpret_cast<const unsigned short *>(A);
    const unsigned short *B_s = reinterpret_cast<const unsigned short *>(B);

    // 16bit * 2 = 32bit씩 묶기 (PTX는 32bit register 단위만 받음)
    unsigned A_pack[4], B_pack[2];
    for (int i = 0; i < 4; ++i)
        A_pack[i] = (A_s[2*i+1] << 16) | A_s[2*i];
    for (int i = 0; i < 2; ++i)
        B_pack[i] = (B_s[2*i+1] << 16) | B_s[2*i];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A_pack[0]), "r"(A_pack[1]), "r"(A_pack[2]), "r"(A_pack[3]),
          "r"(B_pack[0]), "r"(B_pack[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
}

// --- 2. 간단한 커널 -------------------------------------------------------
__global__ void test_mma_kernel(half *A, half *B, half *out, int M, int N, int K)  {
if (blockIdx.x < ceil_div(M, TILE_M) && blockIdx.y < ceil_div(N, TILE_N)) {

    // 각 warp는 16×8×16 타일 하나를 수행한다고 가정
    float D[4], C[4];
    

    // 초기화: A, B 전부 1.0, C는 0.0
   for (int i = 0; i < 4; ++i)  C[i] = 0.0f;

    mma_m16n8k16_f16(D, A, B, C);

    // 결과 저장 (warp 0의 thread 0만 기록)
    if (threadIdx.x == 0)
        for (int i = 0; i < 4; ++i)
            out[i] = __float2half(D[i]); // fp32 누적, fp16으로 변환 
    }

}

// --- 3. main --------------------------------------------------------------
int main() {
    int M = 32, N = 16, K = 16; // 타일 크기

    half *d_A, *d_B, *d_out;
    cudaMalloc(&d_A, sizeof(half)*M*K);
    cudaMalloc(&d_B, sizeof(half)*K*N);
    cudaMalloc(&d_out, sizeof(half)*M*N);

    // 호스트에서 초기화 후 디바이스로 복사
    half *hA = new half[M*K];
    half *hB = new half[K*N];
    for (int i=0;i<M*K;++i) hA[i] = __float2half(1.0f);
    for (int i=0;i<K*N;++i) hB[i] = __float2half(1.0f);
    cudaMemcpy(d_A, hA, sizeof(hA)*K*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, sizeof(hB)*K*N, cudaMemcpyHostToDevice);

    dim3 grid(ceil_div(N, TILE_N), ceil_div(M, TILE_M));  // x: 타일 N방향, y: 타일 M방향
    dim3 block(32, 1, 1);   
    test_mma_kernel<<<grid, block>>>(d_A, d_B, d_out, M, N, K);
    cudaDeviceSynchronize();

    half* h_out = new half[M*N];
    cudaMemcpy(h_out, d_out, sizeof(half) * M*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; ++i)
        printf("D[%d] = %f\n", i, __half2float(h_out[i]));

    cudaFree(d_out);
    return 0;
}
