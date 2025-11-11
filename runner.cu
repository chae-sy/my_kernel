#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"

// --- 매크로 충돌 방지 ---
#ifdef M
#undef M
#endif
#ifdef N
#undef N
#endif
#ifdef K
#undef K
#endif

#include "kernel.h"

// =======================================================
// PyTorch binding wrappers
// =======================================================

// kernel 0.1
void launch_matmul_0_1(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matmul_0_1<<<grid, block>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        M, N, K);
}

// kernel 1.0
void mma_matmul_1_0_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const half *A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half *B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half *C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    dim3 block(32);
    dim3 grid(N/8, M/16);
    mma_matmul_1_0<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
}

// kernel 1.1
void mma_matmul_1_1_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const half *A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half *B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half *C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    const int M_TILE = 2;
    const int N_TILE = 2;
    dim3 block(16, 16);
    dim3 grid((N + N_TILE*32 - 1) / (N_TILE*32),
              (M + M_TILE*32 - 1) / (M_TILE*32));
    mma_matmul_1_1<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
}

// =======================================================
// kernel 2.0 (Permuted shared-memory layout)
// =======================================================
void mma_matmul_2_0_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const half *A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half *B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half *C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    dim3 block(16, 16);
    dim3 grid(N / 64, M / 64);

    mma_matmul_2_0<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
}

// =======================================================
// kernel 2.1 (Permuted layout + A loaded once)
// =======================================================
void mma_matmul_2_1_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const half *A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half *B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half *C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    dim3 block(16, 16);
    dim3 grid(N / 64, M / 64);

    mma_matmul_2_1<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
}

// =======================================================
// PyBind11 module export
// =======================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_0_1", &launch_matmul_0_1, "Naive matmul");
    m.def("mma_matmul_1_0", &mma_matmul_1_0_launcher, "MMA kernel 1.0");
    m.def("mma_matmul_1_1", &mma_matmul_1_1_launcher, "MMA kernel 1.1");
    m.def("mma_matmul_2_0", &mma_matmul_2_0_launcher, "MMA kernel 2.0");
    m.def("mma_matmul_2_1", &mma_matmul_2_1_launcher, "MMA kernel 2.1");
}
