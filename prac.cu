#include <cuda_fp16.h>     
#include <cuda_runtime.h> 
#include <iostream>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

__global__ void mma_fp16_fp32(float *C, half *A, half *B) {
    // Tile 크기: m16n16k16 (16x16 행렬 블록)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // C16×16​=A16×16​×B16×16​, row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag; // col major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag; // fp32

    // C를 0으로 초기화
    wmma::fill_fragment(c_frag, 0.0f);

    // A, B를 global memory에서 fragment로 load
    wmma::load_matrix_sync(a_frag, A, 16); // 16: leading dimension(=stride).
    wmma::load_matrix_sync(b_frag, B, 16); // 16: B는 col_major로 읽으므로, 여기서 16은 열의 길이(행 수).

    // MMA 연산 수행 (FP16 × FP16 → FP32 누적)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // C←A×B+C
    // 중요: warp(32 threads) 단위로 실행되며, m16n16k16 타일 하나를 한 번에 곱-누적.

    // 결과를 global memory에 저장
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major); // 16: 결과 C를 row_major로 저장할 때의 leading dimension(행의 길이=열 수).
}


int main() {
    half *A, *B;
    float *C;
    cudaMallocManaged(&A, sizeof(half) * 16 * 16);
    cudaMallocManaged(&B, sizeof(half) * 16 * 16);
    cudaMallocManaged(&C, sizeof(float) * 16 * 16);

    // 값 초기화
    for (int i = 0; i < 16*16; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
        C[i] = 0.0f;
    }

    mma_fp16_fp32<<<1, 32>>>(C, A, B); // (1, 16) x (16, 1) = (1, 1)
    cudaDeviceSynchronize();

    printf("C[0] = %f\n", C[0]);
    return 0;
}
