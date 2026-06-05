#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// ================================================================
// Do not modify
// ================================================================
#define BLOCK_SIZE        256    // threads per block
#define BLOCK_QUEUE_SIZE  4096   // size of the per-block (shared-memory) queue
#define AVG_DEGREE        8      // out-edges generated per vertex
#define SOURCE            0      // BFS source vertex

void verify(int *label_cpu, int *label_gpu, int num_vertices) {
    printf("Verifying results...\n");
    fflush(stdout);

    long long match_cnt = 0;
    for (int i = 0; i < num_vertices; i++)
        if (label_cpu[i] == label_gpu[i])
            match_cnt++;

    if (match_cnt == num_vertices)
        printf("TEST PASSED\n\n");
    else
        printf("TEST FAILED  (matched %lld / %d)\n\n", match_cnt, num_vertices);
}

void genGraph(int *edges, int *dest, int num_vertices) {
    int idx = 0;
    for (int v = 0; v < num_vertices; v++) {
        edges[v] = idx;
        for (int e = 0; e < AVG_DEGREE; e++)
            dest[idx++] = rand() % num_vertices;
    }
    edges[num_vertices] = idx;
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
//  helper (provided)
// ================================================================
void insert_frontier(int vertex, int *frontier, int *frontier_tail) {
    frontier[*frontier_tail] = vertex;
    (*frontier_tail)++;
}

static void init_device_state(unsigned int source,
                              unsigned int *label_d, unsigned int *visited_d,
                              unsigned int *p_frontier_d,
                              unsigned int *p_frontier_tail_d,
                              unsigned int *c_frontier_tail_d,
                              unsigned int num_vertices)
{
    cudaMemset(visited_d, 0,    num_vertices * sizeof(unsigned int));   // 0 = not visited
    cudaMemset(label_d,   0xFF, num_vertices * sizeof(unsigned int));   // 0xFF bytes -> -1

    unsigned int zero = 0, one = 1, src = source;
    cudaMemcpy(&label_d[source],   &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&visited_d[source], &one,  sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&p_frontier_d[0],   &src,  sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_frontier_tail_d,  &one,  sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_frontier_tail_d,  &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
}


// ================================================================
//  1) SEQUENTIAL BFS  (CPU)
// ================================================================
void BFS_sequential(int source, int *edges, int *dest, int *label, int num_vertices)
{
    for (int i = 0; i < num_vertices; i++)
        label[i] = -1;

    int *frontier = (int*)malloc(num_vertices * sizeof(int));
    if (!frontier) {
        fprintf(stderr, "CPU frontier malloc failed\n");
        exit(EXIT_FAILURE);
    }

    int head = 0;
    int tail = 0;

    label[source] = 0;
    insert_frontier(source, frontier, &tail);

    while (head < tail) {
        int v = frontier[head++];

        for (int e = edges[v]; e < edges[v + 1]; e++) {
            int u = dest[e];

            if (label[u] == -1) {
                label[u] = label[v] + 1;
                insert_frontier(u, frontier, &tail);
            }
        }
    }

    free(frontier);
}


// ================================================================
//  2) PARALLEL BFS kernel  (single GLOBAL queue, no block queue)
// ================================================================
__global__ void BFS_parallel_kernel(unsigned int *p_frontier, unsigned int *p_frontier_tail,
                                     unsigned int *c_frontier, unsigned int *c_frontier_tail,
                                     unsigned int *edges,      unsigned int *dest,
                                     unsigned int *label,      unsigned int *visited)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int frontier_size = *p_frontier_tail;

    if (tid >= frontier_size)
        return;

    unsigned int v = p_frontier[tid];
    unsigned int next_level = label[v] + 1;

    for (unsigned int e = edges[v]; e < edges[v + 1]; e++) {
        unsigned int u = dest[e];

        if (atomicCAS(&visited[u], 0u, 1u) == 0u) {
            label[u] = next_level;
            unsigned int pos = atomicAdd(c_frontier_tail, 1u);
            c_frontier[pos] = u;
        }
    }
}


// ================================================================
//  3) HIERARCHICAL-QUEUE BFS kernel
// ================================================================
__global__ void BFS_Bqueue_kernel(unsigned int *p_frontier, unsigned int *p_frontier_tail,
    unsigned int *c_frontier, unsigned int *c_frontier_tail,
    unsigned int *edges,      unsigned int *dest,
    unsigned int *label,      unsigned int *visited)
{
    
    __shared__ unsigned int block_queue[BLOCK_QUEUE_SIZE];
    __shared__ unsigned int block_tail;
    __shared__ unsigned int global_base;

    if (threadIdx.x == 0)
        block_tail = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int frontier_size = *p_frontier_tail;

    if (tid < frontier_size) {
        unsigned int v = p_frontier[tid];
        unsigned int next_level = label[v] + 1;

        for (unsigned int e = edges[v]; e < edges[v + 1]; e++) {
            unsigned int u = dest[e];

            if (atomicCAS(&visited[u], 0u, 1u) == 0u) {
                label[u] = next_level;

                unsigned int local_pos = atomicAdd(&block_tail, 1u);

                if (local_pos < BLOCK_QUEUE_SIZE) {
                    block_queue[local_pos] = u;
                } else {
                    // Fallback path. With AVG_DEGREE=8 and BLOCK_SIZE=256,
                    // overflow is unlikely, but this keeps the kernel safe.
                    unsigned int global_pos = atomicAdd(c_frontier_tail, 1u);
                    c_frontier[global_pos] = u;
                }
            }
        }
    }

    __syncthreads();

    unsigned int copy_count = block_tail;
    if (copy_count > BLOCK_QUEUE_SIZE)
        copy_count = BLOCK_QUEUE_SIZE;

    if (threadIdx.x == 0)
        global_base = atomicAdd(c_frontier_tail, copy_count);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < copy_count; i += blockDim.x)
        c_frontier[global_base + i] = block_queue[i];
}


// ================================================================
//  Host-side Code for Parallel BFS
// ================================================================
void BFS_parallel(unsigned int source,
                  unsigned int *edges_d, unsigned int *dest_d, unsigned int *label_d,
                  unsigned int *visited_d, unsigned int *frontier_d,
                  unsigned int *c_frontier_tail_d, unsigned int *p_frontier_tail_d,
                  unsigned int num_vertices)
{
    unsigned int *c_frontier_d = &frontier_d[0];
    unsigned int *p_frontier_d = &frontier_d[num_vertices];
    
    init_device_state(source, label_d, visited_d, p_frontier_d,
                       p_frontier_tail_d, c_frontier_tail_d, num_vertices);

    unsigned int p_frontier_tail = 1;
    while (p_frontier_tail > 0) {
       unsigned int zero = 0;
        CUDA_CHECK(cudaMemcpy(c_frontier_tail_d, &zero, sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        unsigned int num_blocks = (p_frontier_tail + BLOCK_SIZE - 1) / BLOCK_SIZE;
        BFS_parallel_kernel<<<num_blocks, BLOCK_SIZE>>>(
            p_frontier_d, p_frontier_tail_d,
            c_frontier_d, c_frontier_tail_d,
            edges_d, dest_d, label_d, visited_d
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&p_frontier_tail, c_frontier_tail_d,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(p_frontier_tail_d, &p_frontier_tail,
                              sizeof(unsigned int), cudaMemcpyHostToDevice));

        unsigned int *tmp = p_frontier_d;
        p_frontier_d = c_frontier_d;
        c_frontier_d = tmp;
    }
}


// ================================================================
//  Host-side Code for Hierarchical BFS
// ================================================================
void BFS_hierarchical(unsigned int source,
                      unsigned int *edges_d, unsigned int *dest_d, unsigned int *label_d,
                      unsigned int *visited_d, unsigned int *frontier_d,
                      unsigned int *c_frontier_tail_d, unsigned int *p_frontier_tail_d,
                      unsigned int num_vertices)
{
    unsigned int *c_frontier_d = &frontier_d[0];
    unsigned int *p_frontier_d = &frontier_d[num_vertices];

    init_device_state(source, label_d, visited_d, p_frontier_d,
                       p_frontier_tail_d, c_frontier_tail_d, num_vertices);

    unsigned int p_frontier_tail = 1;
    while (p_frontier_tail > 0) {
        unsigned int zero = 0;
        CUDA_CHECK(cudaMemcpy(c_frontier_tail_d, &zero, sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        unsigned int num_blocks = (p_frontier_tail + BLOCK_SIZE - 1) / BLOCK_SIZE;
        BFS_Bqueue_kernel<<<num_blocks, BLOCK_SIZE>>>(
            p_frontier_d, p_frontier_tail_d,
            c_frontier_d, c_frontier_tail_d,
            edges_d, dest_d, label_d, visited_d
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&p_frontier_tail, c_frontier_tail_d,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(p_frontier_tail_d, &p_frontier_tail,
                              sizeof(unsigned int), cudaMemcpyHostToDevice));

        unsigned int *tmp = p_frontier_d;
        p_frontier_d = c_frontier_d;
        c_frontier_d = tmp;
    }
}


// ================================================================
//  MAIN
// ================================================================
int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./bfs <num_vertices>\n\n");
        exit(0);
    }

    int num_vertices = atoi(argv[1]);
    int num_edges    = num_vertices * AVG_DEGREE;

    int *edges     = (int*)malloc((num_vertices + 1) * sizeof(int));
    int *dest      = (int*)malloc(num_edges          * sizeof(int));
    int *Label_CPU = (int*)malloc(num_vertices       * sizeof(int));
    int *Label_GPU = (int*)malloc(num_vertices       * sizeof(int));

    if (!edges || !dest || !Label_CPU || !Label_GPU) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    genGraph(edges, dest, num_vertices);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    unsigned int *edges_d, *dest_d, *label_d, *visited_d, *frontier_d;
    unsigned int *c_frontier_tail_d, *p_frontier_tail_d;

    // ======================= INSERT CODE HERE =======================
    //  DEVICE MEMORY ALLOCATION  (cudaMalloc, unsigned int)
    //    edges_d           (num_vertices + 1)
    //    dest_d            (num_edges)
    //    label_d           (num_vertices)
    //    visited_d         (num_vertices)
    //    frontier_d        (2 * num_vertices)   // current + previous
    //    c_frontier_tail_d (1)
    //    p_frontier_tail_d (1)
    // ================================================================
    CUDA_CHECK(cudaMalloc((void**)&edges_d, (num_vertices + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&dest_d, num_edges * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&label_d, num_vertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&visited_d, num_vertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&frontier_d, 2 * num_vertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&c_frontier_tail_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&p_frontier_tail_d, sizeof(unsigned int)));


    // ======================= INSERT CODE HERE =======================
    //  MEMORY COPY (Host -> Device): copy edges, dest into edges_d, dest_d
    // ================================================================
    CUDA_CHECK(cudaMemcpy(edges_d, edges, (num_vertices + 1) * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dest_d, dest, num_edges * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));


    // ---- (1) CPU sequential BFS : reference + CPU timing (provided) ----
    struct timespec cpu_start, cpu_stop;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);

    BFS_sequential(SOURCE, edges, dest, Label_CPU, num_vertices);

    clock_gettime(CLOCK_MONOTONIC, &cpu_stop);
    float cpu_time = (cpu_stop.tv_sec  - cpu_start.tv_sec)  * 1000.0f
                   + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1.0e6f;
    printf("[Sequential   CPU] elapsed time = %.3f msec\n", cpu_time);

    // ---- (2) Parallel BFS : GPU timing + verify ----
    CUDA_CHECK(cudaEventRecord(start, 0));

    BFS_parallel(SOURCE, edges_d, dest_d, label_d, visited_d, frontier_d,
                 c_frontier_tail_d, p_frontier_tail_d, num_vertices);
    CUDA_CHECK(cudaMemcpy(Label_GPU, label_d, num_vertices * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    float time_par;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_par, start, stop));
    printf("[Parallel     GPU] elapsed time = %.3f msec\n", time_par);
    verify(Label_CPU, Label_GPU, num_vertices);

    // ---- (3) Hierarchical-queue BFS : GPU timing + verify ----
    CUDA_CHECK(cudaEventRecord(start, 0));

    BFS_hierarchical(SOURCE, edges_d, dest_d, label_d, visited_d, frontier_d,
                     c_frontier_tail_d, p_frontier_tail_d, num_vertices);
    CUDA_CHECK(cudaMemcpy(Label_GPU, label_d, num_vertices * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    float time_hq;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_hq, start, stop));
    printf("[Hierarchical GPU] elapsed time = %.3f msec\n", time_hq);
    verify(Label_CPU, Label_GPU, num_vertices);

    fflush(stdout);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ======================= INSERT CODE HERE =======================
    //  FREE ALLOCATED MEMORY  (7 device pointers via cudaFree,
    //                          4 host pointers via free)
    // ================================================================
    CUDA_CHECK(cudaFree(edges_d));
    CUDA_CHECK(cudaFree(dest_d));
    CUDA_CHECK(cudaFree(label_d));
    CUDA_CHECK(cudaFree(visited_d));
    CUDA_CHECK(cudaFree(frontier_d));
    CUDA_CHECK(cudaFree(c_frontier_tail_d));
    CUDA_CHECK(cudaFree(p_frontier_tail_d));

    free(edges);
    free(dest);
    free(Label_CPU);
    free(Label_GPU);
    return 0;
}
