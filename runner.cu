#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>
#include <cublas_v2.h>
#include <type_traits>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>

#include "kernel.h"

#define M 256
#define N 256
#define K 256

#define WARMUP_REPS 50
#define REPS        100

void check_cublas(cublasStatus_t stat, const char* const func,
                  const char* const file, const int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        const char* err_str = nullptr;
        switch (stat) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                err_str = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                err_str = "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE:
                err_str = "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                err_str = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                err_str = "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                err_str = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                err_str = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            default:
                err_str = "Unknown cuBLAS error"; break;
        }

        fprintf(stderr,
                "cuBLAS error at %s:%d\n"
                "    -> Function: %s\n"
                "    -> Error: %s (%d)\n",
                file, line, func, err_str, static_cast<int>(stat));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS(val) check_cublas((val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr,
                "CUDA error at %s:%d\n"
                "    -> Function: %s\n"
                "    -> Error: %s (%d)\n",
                file, line, func, cudaGetErrorString(err), static_cast<int>(err));
        // 즉시 종료
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda_last(const char* const file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "CUDA kernel launch error at %s:%d\n"
                "    -> Error: %s (%d)\n",
                file, line, cudaGetErrorString(err), static_cast<int>(err));
        exit(EXIT_FAILURE);
    }
}
#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)

#define ceilDiv(x, y) (((x) + (y) - 1) / (y))

static inline bool in_array(int val, const int *vals, int n) {
  for (int i = 0; i < n; i++) if (vals[i] == val) return true;
  return false;
}

// N(0,1) 분포의 FP16 행렬을 row-major 형태로 초기화
void random_init_matrix_half(half* A, size_t m, size_t n)
{
    std::default_random_engine eng(0U);
    std::normal_distribution<float> dis(0.0f, 1.0f); // N(0,1)
    auto const randn = [&dis, &eng]() { return dis(eng); };
    for (size_t i=0; i < m; ++i)
    {
        for (size_t j=0; j < n; ++j)
        {
            A[i * n + j] = __float2half(randn());
        }
    }
}

void gemm_cpu(const half *A, const half *B, half *C,
                      size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.f; // FP32 accumulation
            for (size_t l = 0; l < k; ++l) {
                half a = A[i*k + l];
                half b = B[l*n + j];
                half prod_h = __hmul(a, b);      // FP16 × FP16 = FP16 
                
                acc += __half2float(prod_h);       // 누산은 FP32
            }
            C[i*n + j] = __float2half(acc); // FP32 → FP16
        }
    }
}

// cublas launcher
static inline void launch_cublas_kernel(
    cublasHandle_t h, half *dA, half *dB, half *dC,
    int m, int n, int k) {
  const float alpha = 1.f, beta = 0.f;
  CHECK_CUBLAS(cublasGemmEx(
      h,
      CUBLAS_OP_T, CUBLAS_OP_T,
      n, m, k,
      &alpha,
      dB, CUDA_R_16F, n,
      dA, CUDA_R_16F, k,
      &beta,
      dC, CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


// kernel launcher
static inline void launch_custom_kernel(
    int kernelNum, dim3 grid, dim3 block,
    const half *A, const half *B, half *C, int m, int n, int k, cudaStream_t stream) {
  switch (kernelNum) {
    case 1: matmul_0_1<<<grid, block, 0, stream>>>(A, B, C, m, n, k); break;
    //case 11: mma_matmul_1_1<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 20: mma_matmul_2_0<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 21: mma_matmul_2_1<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 30: mma_matmul_3_0<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 31: mma_matmul_3_1<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 32: mma_matmul_3_2<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 33: mma_matmul_3_3<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    //case 34: mma_matmul_3_4<<<grid, block>>>(A, B_colmajor, C, M, N, K); break;
    default: break;
  }
}

// timing
template <typename T>
static inline float time_kernel_ms(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_warmups, size_t num_repeats
                          )
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}
int main(int argc, char **argv){
    if (argc != 2) {
    printf("Usage: ./runner <kernelNum>\n");
    printf("Valid: 1,10,11,20,21,30,31,32,33,34\n");
    return 0;
  }
    int kernelNum = atoi(argv[1]);
  const int valid[11] = {1,10,11,20,21,30,31,32,33,34};
  if (!in_array(kernelNum, valid, 11)) {
    printf("Invalid kernel.\n"); return 0;
  }
  printf("Running Kernel %d.%d\n", kernelNum/10, kernelNum%10);

    half *hA, *hB, *hC, *ref_C;
    hA = new half[M * K](); // set to 0
    hB = new half[K * N]();
    hC = new half[M * N]();
    ref_C = new half[M * N]();

    half *dA, *dB, *dC, *dC_cublas, *dC_custom;
    cudaMalloc(&dA, M * K * sizeof(half));
    cudaMalloc(&dB, K * N * sizeof(half));
    cudaMalloc(&dC, M * N * sizeof(half));
    cudaMalloc(&dC_cublas, M * N * sizeof(half));
    cudaMalloc(&dC_custom, M * N * sizeof(half));

    random_init_matrix_half(hA, M, K);
    random_init_matrix_half(hB, K, N);

    // accuracy check vs gemm cpu
    cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(16, 16); 
    dim3 grid(ceilDiv(N, block.x) , ceilDiv(M, block.y)); 
    cudaStream_t stream;
CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    launch_custom_kernel(kernelNum, grid, block, dA, dB, dC, M, N, K, stream);
CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    cudaDeviceSynchronize(); // GPU 연산이 모두 끝날 때까지 기다리기

    cudaMemcpy(hC, dC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    gemm_cpu(hA, hB, ref_C, M, N, K);
    const double abs_tol = 5.0e-2, rel_tol = 2.0e-2;
    double avg_abs=0, avg_diff=0, max_abs=0, avg_out=0;
    int mismatch=0, total = M*N;
    for (size_t i=0; i<total; i++){
        double ref  = __half2float(ref_C[i]);
        double test = __half2float(hC[i]);
        double diff = test - ref;
        double abs_diff   = fabs(diff);
        if (abs_diff>abs_tol && abs_diff/(fabs(ref)+1e-8)>rel_tol) mismatch++;
        avg_abs += abs_diff; avg_diff += diff; avg_out += test;
        if (abs_diff > max_abs) max_abs = abs_diff;
    }
    avg_abs  /= total; avg_diff /= total; avg_out /= total;

    printf("\n=== Accuracy ===\n");
    printf("max abs diff %.6f, avg abs diff %.6f, avg diff %.6f, avg out %.6f\n",
            (float)max_abs, (float)avg_abs, (float)avg_diff, (float)avg_out);
    printf("mismatches (>%.1e or >%.1e): %d / %d\n", abs_tol, rel_tol, mismatch, total);

    // latency check
    // auto fn_cublas = [&](cudaStream_t stream){
    // launch_cublas_kernel(dA, dB, dC_cublas, M, N, K, cudaStream_t stream);
    // };
    // auto fn_custom = [&](cudaStream_t stream){
    //     launch_custom_kernel(kernelNum, grid, block, dA, dB, dC_custom, M, N, K, cudaStream_t stream);
    // };
    // float t_cublas = time_kernel_ms(fn_cublas, WARMUP_REPS, REPS);
    // float t_custom    = time_kernel_ms(fn_custom, WARMUP_REPS, REPS);
    cudaStream_t stream2;
CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
cublasHandle_t handle;
CHECK_CUBLAS(cublasCreate(&handle));
CHECK_CUBLAS(cublasSetStream(handle, stream2));
float const t_cublas = time_kernel_ms<void>(
  [&](cudaStream_t s){
    // 여기서는 핸들 재사용. 절대 DeviceSynchronize() 넣지 마!
    launch_cublas_kernel(handle, dA, dB, dC_cublas, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
  }, stream2, WARMUP_REPS, REPS);

float const t_custom = time_kernel_ms<void>(
  [&](cudaStream_t s){
    launch_custom_kernel(kernelNum, grid, block, dA, dB, dC_custom, M, N, K, s);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
  }, stream2, WARMUP_REPS, REPS);

CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));




    // gflops & ratio
    const double ops = 2.0 * (double)M * (double)N * (double)K / 1e9;
    const double gflops_cublas  = ops / (t_cublas / 1e3);
    const double gflops_custom = ops / (t_custom    / 1e3);
    const double perf_ratio = (gflops_custom / gflops_cublas) * 100.0;

    printf("\n=== Performance (warmup %d + iters %d) ===\n", WARMUP_REPS, REPS);
    printf("cuBLAS: %.3f ms  |  %.2f GFLOP/s\n", t_cublas, gflops_cublas);
    printf("custom   : %.3f ms  |  %.2f GFLOP/s\n", t_custom,    gflops_custom);
    printf("custom / cuBLAS: %.2f%%\n", perf_ratio);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}