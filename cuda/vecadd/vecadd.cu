#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *A_d, int* B_d, int* C_d, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n) C_d[i] = A_d[i]+B_d[i];
}

void vecAdd(int* A, int* B, int* C, int n)
{
    int size = n * sizeof(int);
    int* A_d = 0;
    int* B_d = 0;
    int* C_d = 0;
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);
    
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    addKernel<<<(n + 255) / 256,256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main(void)
{
    const int SIZE = 2048; // 총 2048개 thread. 8개 블럭. 각 블럭 당 256 thread. 
    int a[SIZE], b[SIZE], c[SIZE];

    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = i;
    }

    vecAdd(a, b, c, SIZE);

    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}