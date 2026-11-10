#include <stdio.h>
#include <stdlib.h>

#define GRIDSIZE (32*1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

__global__ void kernel(unsigned long long int* pCount){
    __shared__ int nCountShared;
    if (threadIdx.x == 0){
        nCountShared = 0;
    }
    __syncthreads();
    atomicAdd(&nCountShared, 1);
    __syncthreads();
    if (threadIdx.x==0){
        atomicAdd(pCount, nCountShared);
    }
}

int main(void){
    unsigned long long int aCount[1];
    
    // prepare a timer
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CUDA: allocate device memory
    unsigned long long int* pCountDev = NULL;
    cudaMalloc((void**)&pCountDev, sizeof(unsigned long long int));
    cudaMemset(pCountDev, 0, sizeof(unsigned long long int));
    
    // start timer
    cudaEventRecord(start, 0);
    
    // CUDA: launch the kernel
    dim3 dimGrid(GRIDSIZE, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);
    kernel<<<dimGrid, dimBlock>>>(pCountDev);
    
    // CUDA: copy from device to host
    cudaMemcpy(aCount, pCountDev, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    printf("total number of threads = %llu\n", TOTALSIZE);
    printf("count = %llu\n", aCount[0]);
    
    // end timer
    float time;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("elased time = %f msec\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // CUDA: free the memory
    cudaFree(pCountDev);
}