#include <stdio.h>
#include <stdlib.h>

#define GRIDSIZE 1024
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)
#define NUMHIST 16

void genData(unsigned int* ptr, unsigned int size){
    while (size--){
        *ptr++=(unsigned int)(rand()%(NUMHIST-1));
    }
}

__global__ void kernel(unsigned int* hist, unsigned int* img, unsigned int size){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelVal = img[i];
    atomicAdd(&(hist[pixelVal]), 1);
}

int main (void){
    unsigned int* pImage = NULL;
    unsigned int* pHistogram = NULL;
    int i;

    // prepare a timer
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // malloc memories on the host-side
    pImage = (unsigned int*)malloc(TOTALSIZE*sizeof(unsigned int));
    pHistogram = (unsigned int*)malloc(NUMHIST * sizeof(unsigned int));

    //generate source data
    genData(pImage, TOTALSIZE);

    //CUDA: allocate device memory
    unsigned int* pImageDev;
    unsigned int* pHistogramDev;
    cudaMalloc((void**)&pImageDev, TOTALSIZE*sizeof(unsigned int));
    cudaMalloc((void**)&pHistogramDev, NUMHIST*sizeof(unsigned int));
    cudaMemset(pHistogramDev, 0, NUMHIST*sizeof(unsigned int));

    // CUDA: copy from host to device
    cudaMemcpy(pImageDev, pImage, TOTALSIZE*sizeof(unsigned int), cudaMemcpyHostToDevice);

    //start the timer
    cudaEventRecord(start, 0);
    //perform the action
    dim3 dimGrid(GRIDSIZE, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);
    kernel<<<dimGrid, dimBlock>>>(pHistogramDev, pImageDev, TOTALSIZE);

    //end the timer
    float time;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("elased time = %f msec\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CUDA: copy from device to host
    cudaMemcpy(pHistogram, pHistogramDev, NUMHIST*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //print the histogram
    long total = 0L;
    for (i=0; i<NUMHIST; ++i){
        printf("%2d: %10d\n", i, pHistogram[i]);
        total += pHistogram[i];
    }
    printf("total: %10ld (should be %ld)\n", total, TOTALSIZE);

    //CUDA: free the memory
    cudaFree(pImageDev);
    cudaFree(pHistogramDev);

    // free the memory
    free(pImage);
    free(pHistogram);
}