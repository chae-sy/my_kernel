#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 256
#define BLOCK_QUEUE_SIZE 1024
#define MAX_FRONTIER_SIZE 1000000

__global__ void BFS_init_kernel(
    unsigned int source,
    unsigned int *label,
    unsigned int *visited,
    unsigned int *p_frontier,
    unsigned int *p_frontier_tail,
    unsigned int *c_frontier_tail,
    unsigned int num_vertices
){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_vertices) {
        visited[tid] = 0;
        label[tid] = UINT_MAX;
    }

    if (tid == 0) {
        visited[source] = 1;
        label[source] = 0;
        p_frontier[0] = source;
        *p_frontier_tail = 1;
        *c_frontier_tail = 0;
    }
}

__global__ void reset_tail_kernel(
    unsigned int *p_frontier_tail,
    unsigned int *c_frontier_tail,
    unsigned int new_tail
){
    *p_frontier_tail = new_tail;
    *c_frontier_tail = 0;
}

__global__ void BFS_Bqueue_kernel(unsigned int *p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier, unsigned int *c_frontier_tail, unsigned int *edge, unsigned int *dest, unsigned int *label, unsigned int *visited)
{
    __shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
    __shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;

    if (threadIdx.x == 0) c_frontier_tail_s = 0;
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *p_frontier_tail)
    {
        const unsigned int my_vertex = p_frontier[tid];
        for(unsigned int i=edge[my_vertex]; i<edge[my_vertex+1]; i++)
        {
           const unsigned int was_visited = atomicExch(&visited[dest[i]], 1);
              if (!was_visited){
                label[dest[i]] = label[my_vertex] +1;
                const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1);
                if (my_tail < BLOCK_QUEUE_SIZE){
                    c_frontier_s[my_tail] = dest[i];
                }
                else{
                    c_frontier_tail_s = BLOCK_QUEUE_SIZE;
                    const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
                    c_frontier[my_global_tail] = dest[i];
                }
              }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0){
        our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();
    for(unsigned int i=threadIdx.x; i<c_frontier_tail_s; i+=blockDim.x){
        c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
}


void BFS_host(
    unsigned int source,
    unsigned int *edge,
    unsigned int *dest,
    unsigned int *label,
    unsigned int num_vertices,
    unsigned int num_edges
){
    unsigned int *edge_d, *dest_d, *label_d;
    unsigned int *visited_d;
    unsigned int *frontier_d;
    unsigned int *p_frontier_tail_d, *c_frontier_tail_d;

    cudaMalloc(&edge_d, sizeof(unsigned int) * (num_vertices + 1));
    cudaMalloc(&dest_d, sizeof(unsigned int) * num_edges);
    cudaMalloc(&label_d, sizeof(unsigned int) * num_vertices);
    cudaMalloc(&visited_d, sizeof(unsigned int) * num_vertices);

    cudaMemcpy(edge_d, edge, sizeof(unsigned int) * (num_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dest_d, dest, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice);

    cudaMalloc(&frontier_d, sizeof(unsigned int) * 2 * MAX_FRONTIER_SIZE);
    cudaMalloc(&p_frontier_tail_d, sizeof(unsigned int));
    cudaMalloc(&c_frontier_tail_d, sizeof(unsigned int));

    unsigned int *c_frontier_d = &frontier_d[0];
    unsigned int *p_frontier_d = &frontier_d[MAX_FRONTIER_SIZE];

    int init_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    BFS_init_kernel<<<init_blocks, BLOCK_SIZE>>>(
        source,
        label_d,
        visited_d,
        p_frontier_d,
        p_frontier_tail_d,
        c_frontier_tail_d,
        num_vertices
    );

    unsigned int p_frontier_tail = 1;

    while (p_frontier_tail > 0) {
        int num_blocks = (p_frontier_tail + BLOCK_SIZE - 1) / BLOCK_SIZE;

        BFS_Bqueue_kernel<<<num_blocks, BLOCK_SIZE>>>(
            p_frontier_d,
            p_frontier_tail_d,
            c_frontier_d,
            c_frontier_tail_d,
            edge_d,
            dest_d,
            label_d,
            visited_d
        );

        cudaMemcpy(
            &p_frontier_tail,
            c_frontier_tail_d,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        );

        unsigned int *temp = c_frontier_d;
        c_frontier_d = p_frontier_d;
        p_frontier_d = temp;

        reset_tail_kernel<<<1, 1>>>(
            p_frontier_tail_d,
            c_frontier_tail_d,
            p_frontier_tail
        );
    }

    cudaMemcpy(label, label_d, sizeof(unsigned int) * num_vertices, cudaMemcpyDeviceToHost);

    cudaFree(edge_d);
    cudaFree(dest_d);
    cudaFree(label_d);
    cudaFree(visited_d);
    cudaFree(frontier_d);
    cudaFree(p_frontier_tail_d);
    cudaFree(c_frontier_tail_d);
}



int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s graph.txt\n", argv[0]);
        return -1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Cannot open %s\n", argv[1]);
        return -1;
    }

    unsigned int num_vertices;
    unsigned int num_edges;

    fscanf(fp, "%u %u", &num_vertices, &num_edges);

    unsigned int *edge =
        (unsigned int *)malloc(sizeof(unsigned int) * (num_vertices + 1));

    unsigned int *dest =
        (unsigned int *)malloc(sizeof(unsigned int) * num_edges);

    unsigned int *label =
        (unsigned int *)malloc(sizeof(unsigned int) * num_vertices);

    for (unsigned int i = 0; i < num_vertices; i++)
        label[i] = UINT_MAX;

    /*
        CSR format

        edge[0] edge[1] ... edge[V]
        dest[0] dest[1] ... dest[E-1]

        Example input:

        5 6
        0 2 4 5 6 6
        1 2 0 3 4 2
    */

    for (unsigned int i = 0; i < num_vertices + 1; i++)
        fscanf(fp, "%u", &edge[i]);

    for (unsigned int i = 0; i < num_edges; i++)
        fscanf(fp, "%u", &dest[i]);

    fclose(fp);

    unsigned int source = 0;

    BFS_host(
        source,
        edge,
        dest,
        label,
        num_vertices,
        num_edges
    );

    printf("===== BFS Result =====\n");

    for (unsigned int i = 0; i < num_vertices; i++) {
        if (label[i] == UINT_MAX)
            printf("Vertex %u : unreachable\n", i);
        else
            printf("Vertex %u : level %u\n", i, label[i]);
    }

    free(edge);
    free(dest);
    free(label);

    return 0;
}