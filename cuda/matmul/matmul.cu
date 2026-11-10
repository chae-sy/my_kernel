#include <iostream>
#include <cuda_runtime.h>


__global__ void mulKernel (int* c, const int* a, const int *b, const int WIDTH){
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = y * WIDTH + x;
    int sum = 0;
    for (int k = 0; k<WIDTH; ++k){
        sum += a[y*WIDTH+k]*b[k*WIDTH+x];
    }
    c[i]=sum;
}

int main(void){
    const int WIDTH=5;
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
    dim3 dimBlock(WIDTH, WIDTH, 1); // x, y, z
    mulKernel<<<1, dimBlock>>>(dev_c, dev_a, dev_b, WIDTH);
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