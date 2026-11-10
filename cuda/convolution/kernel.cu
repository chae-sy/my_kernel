#define TILE_SIZE 16
#define RADIUS (FILTER_SIZE / 2)
#define SHARED_SIZE (TILE_SIZE + FILTER_SIZE - 1)

__constant__ float M_c[FILTER_SIZE * FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    __shared__ float N_s[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    for (int y = ty; y < SHARED_SIZE; y += TILE_SIZE) {
        for (int x = tx; x < SHARED_SIZE; x += TILE_SIZE) {

            int globalRow = blockIdx.y * TILE_SIZE + y - RADIUS;
            int globalCol = blockIdx.x * TILE_SIZE + x - RADIUS;

            if (globalRow >= 0 && globalRow < N.height &&
                globalCol >= 0 && globalCol < N.width) {
                N_s[y][x] = N.elements[globalRow * N.pitch + globalCol];
            } else {
                N_s[y][x] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (row < P.height && col < P.width) {
        float sum = 0.0f;

        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                sum += M_c[i * FILTER_SIZE + j] *
                       N_s[ty + i][tx + j];
            }
        }

        P.elements[row * P.pitch + col] = sum;
    }
}