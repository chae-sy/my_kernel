#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 2

__global__ void matmul(int* g_C, const int* g_A, const int* g_B, const int width){
    __shared__ int s_A [TILE_WIDTH][TILE_WIDTH];
    __shared__ int s_B [TILE_WIDTH][TILE_WIDTH];
    int gy = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int gx = blockIdx.x*TILE_WIDTH + threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    float sum = 0;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (register int m = 0; m<numTiles; ++m){
        s_A[ty][tx] = g_A[gy*width + m*TILE_WIDTH+tx];
        s_B[ty][tx] = g_B[(m*TILE_WIDTH+ty)*width + gx];
        __syncthreads();
        for (register int k = 0; k<TILE_WIDTH; ++k){
            sum += s_A[ty][k]*s_B[k][tx];
        }
        __syncthreads();
    }
    g_C[gy*width + gx]= sum;
}


int main(void){
    const int WIDTH=4;
    int a[WIDTH][WIDTH];
    int b[WIDTH][WIDTH];
    int c[WIDTH][WIDTH]= {0};
    for (int y = 0; y<WIDTH; ++y){
        for (int x=0; x<WIDTH; ++x){
            a[y][x] = y*10+x;
            b[y][x] = (y*10+x)*100;
        }
    }
    int *dev_a=0;
    int *dev_b=0;
    int *dev_c=0;
    cudaMalloc((void**)&dev_a, WIDTH*WIDTH*sizeof(int));
    cudaMalloc((void**)&dev_b, WIDTH*WIDTH*sizeof(int));
    cudaMalloc((void**)&dev_c, WIDTH*WIDTH*sizeof(int));
    cudaMemcpy(dev_a, a, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid((WIDTH+TILE_WIDTH-1)/TILE_WIDTH, (WIDTH+TILE_WIDTH-1)/TILE_WIDTH, 1); // x, y, z
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matmul<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, WIDTH);
    cudaMemcpy(c, dev_c, WIDTH*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c); cudaFree(dev_a); cudaFree(dev_b);

    for (int y = 0; y<WIDTH; ++y){
        for (int x= 0; x<WIDTH; ++x){
            printf("%5d ", c[y][x]);
        }
        printf("\n");
    }
    return 0;
}