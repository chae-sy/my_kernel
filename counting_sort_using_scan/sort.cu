#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAX_VALUE       127               // input values are in [1, 127]
#define HIST_SIZE       (MAX_VALUE + 1)   // indices 0..127
#define BLOCK_SIZE      256
#define SCAN_BLOCK_SIZE 128               // next power-of-2 >= HIST_SIZE (128)


// ======================= INSERT CODE HERE =======================
void sort(int* input, int* output, long long n) {
    int hist[HIST_SIZE] = {0};
    int offset[HIST_SIZE] = {0};

    for (long long i = 0; i < n; i++)
        hist[input[i]]++;

    offset[0] = 0;
    for (int i = 1; i < HIST_SIZE; i++)
        offset[i] = offset[i - 1] + hist[i - 1];

    for (long long i = 0; i < n; i++) {
        int v = input[i];
        output[offset[v]++] = v;
    }
}
// ================================================================


// ======================= INSERT CODE HERE =======================
__global__ void histogram_kernel(const int* __restrict__ input,
                                  long long n,
                                  int* hist) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int v = input[idx];
        atomicAdd(&hist[v], 1);
    }
}

__global__ void scan_kernel(const int* hist, int* offset, int size) {
    __shared__ int temp[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;

    if (tid < size)
        temp[tid] = hist[tid];
    else
        temp[tid] = 0;

    __syncthreads();

    for (int stride = 1; stride < SCAN_BLOCK_SIZE; stride *= 2) {
        int val = 0;
        if (tid >= stride)
            val = temp[tid - stride];

        __syncthreads();

        temp[tid] += val;

        __syncthreads();
    }

    if (tid == 0)
        offset[0] = 0;
    else if (tid < size)
        offset[tid] = temp[tid - 1];
}

__global__ void scatter_kernel(const int* __restrict__ input,
                                long long n,
                                int* offset,
                                int* output) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int v = input[idx];
        int pos = atomicAdd(&offset[v], 1);
        output[pos] = v;
    }
}
// ================================================================


// ================================================================
//  UTILITY  (provided – do not modify)
// ================================================================
void verify(int* result_cpu, int* result_gpu, long long input_size) {
    printf("Verifying results...\n");
    fflush(stdout);

    long long match_cnt = 0;
    for (long long i = 0; i < input_size; i++)
        if (result_cpu[i] == result_gpu[i])
            match_cnt++;

    if (match_cnt == input_size)
        printf("TEST PASSED\n\n");
    else
        printf("TEST FAILED  (matched %lld / %lld)\n\n", match_cnt, input_size);
}

void genData(int* ptr, long long size) {
    while (size--)
        *ptr++ = (int)(rand() % MAX_VALUE + 1);   // values in [1, 127]
}

#ifdef DEBUG
#define CUDA_CHECK(x) do {                                      \
    (x);                                                        \
    cudaError_t e = cudaGetLastError();                         \
    if (cudaSuccess != e) {                                     \
        printf("cuda failure \"%s\" at %s:%d\n",                \
               cudaGetErrorString(e), __FILE__, __LINE__);      \
        exit(1);                                                \
    }                                                           \
} while (0)
#else
#define CUDA_CHECK(x) (x)
#endif


// ================================================================
//  MAIN
// ================================================================
int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./sort <input_size>\n\n");
        exit(0);
    }

    long long input_size = (long long)atoll(argv[1]);

    // ---- Host memory allocation ----
    int* Source     = (int*)malloc(input_size * sizeof(int));
    int* Result_CPU = (int*)malloc(input_size * sizeof(int));
    int* Result_GPU = (int*)malloc(input_size * sizeof(int));

    if (!Source || !Result_CPU || !Result_GPU) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    genData(Source, input_size);

    // ---- CUDA timer setup ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // ======================= INSERT CODE HERE =======================
    int *d_Source, *d_Result, *d_hist, *d_offset;

    CUDA_CHECK(cudaMalloc((void**)&d_Source, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_Result, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_hist, HIST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_offset, HIST_SIZE * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_offset, 0, HIST_SIZE * sizeof(int)));
		// ================================================================


    // ======================= INSERT CODE HERE =======================
    CUDA_CHECK(cudaMemcpy(d_Source, Source,
                      input_size * sizeof(int),
                      cudaMemcpyHostToDevice));
    // ================================================================

    CUDA_CHECK(cudaEventRecord(start, 0));


    // ======================= INSERT CODE HERE =======================
    int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    histogram_kernel<<<num_blocks, BLOCK_SIZE>>>(d_Source, input_size, d_hist);
    CUDA_CHECK(cudaDeviceSynchronize());

    scan_kernel<<<1, SCAN_BLOCK_SIZE>>>(d_hist, d_offset, HIST_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(d_Source, input_size, d_offset, d_Result);
    CUDA_CHECK(cudaDeviceSynchronize());
    // ================================================================


    // ======================= INSERT CODE HERE =======================
    CUDA_CHECK(cudaMemcpy(Result_GPU, d_Result,
                      input_size * sizeof(int),
                      cudaMemcpyDeviceToHost));
    // ================================================================




    // ---- GPU timing ----
    float time;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("GPU elapsed time = %.3f msec\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---- CPU sort + timing ----
    struct timespec cpu_start, cpu_stop;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);

    sort(Source, Result_CPU, input_size);

    clock_gettime(CLOCK_MONOTONIC, &cpu_stop);
    float cpu_time = (cpu_stop.tv_sec  - cpu_start.tv_sec)  * 1000.0f
                   + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1.0e6f;
    printf("CPU elapsed time = %.3f msec\n", cpu_time);

    // ---- Verification ----
    verify(Result_CPU, Result_GPU, input_size);
    fflush(stdout);


    // ======================= INSERT CODE HERE =======================
    CUDA_CHECK(cudaFree(d_Source));
    CUDA_CHECK(cudaFree(d_Result));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_offset));

    free(Source);
    free(Result_CPU);
    free(Result_GPU);
    // ================================================================

    return 0;
}
