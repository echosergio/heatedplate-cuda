#define M 500
#define N 500
#define NUM_ELEMENTS M * N
#define DIM_GRID 256
#define DIM_BLOCK 1024

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(msg) (checkCUDAError(msg, __FILE__, __LINE__))

static void checkCUDAError(const char *msg, const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s: %s. In %s at line %d\n", msg, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void copy_grid(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < M && y < N)
        d_u[x + y * N] = d_w[x + y * N];

    __syncthreads();

    return;
}

__device__ double d_epsilon;

__device__ double d_epsilon_reduction[NUM_ELEMENTS];

__device__ double d_epsilon_reduction_results[DIM_BLOCK];

__global__ void epsilon_reduction(double *d_w, double *d_u)
{
    __shared__ double local_reduction[DIM_BLOCK];

    int stride = blockDim.x * gridDim.x;

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int local_index = threadIdx.x;

    local_reduction[local_index] = fabs(d_w[index] - d_u[index]); 

    if ((index + stride) < NUM_ELEMENTS && local_reduction[local_index] < fabs(d_w[index + stride] - d_u[index + stride]))
        local_reduction[local_index] = fabs(d_w[index + stride] - d_u[index + stride]);

    __syncthreads();

    for (int i = blockDim.x>>1; i>0; i>>=1)
    {
        if (local_index < i && local_reduction[local_index] < local_reduction[local_index + i])
            local_reduction[local_index] = local_reduction[local_index + i];

        __syncthreads();
    }

    if(local_index == 0) 
        d_epsilon_reduction_results[blockIdx.x] = local_reduction[local_index];

    return;
}

__global__ void epsilon_reduction_results()
{
    __shared__ double local_reduction[DIM_BLOCK];

    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < blockDim.x)
    {
        local_reduction[index] = 0;
        __syncthreads();

        local_reduction[index] = d_epsilon_reduction_results[index];
        __syncthreads();

        for (int i = blockDim.x>>1; i>0; i>>=1)
        {
            if (index < i && local_reduction[index] < local_reduction[index + i])
                local_reduction[index] = local_reduction[index + i];

            __syncthreads();
        }

        if (index == 0)
            d_epsilon = local_reduction[index];
        __threadfence();
    }

    return;
}

__global__ void calculate_solution(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > 0 && y > 0 && x < M - 1 && y < N - 1)
    {
        int index = x + y * N;

        int west = (x - 1) + y * N;
        int east = (x + 1) + y * N;
        int north = x + (y - 1) * N;
        int south = x + (y + 1) * N;

        d_w[index] = (d_u[north] + d_u[south] + d_u[east] + d_u[west]) / 4.0;
    }

    __syncthreads();

    return;
}

void calculate_solution_kernel(double w[M][N], double epsilon)
{
    double diff;
    int iterations;
    int iterations_print;
    float ElapsedTime;
    cudaEvent_t cudaStart, cudaStop;

    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    const unsigned int matrix_mem_size = sizeof(double) * M * N;

    double *d_w = (double *)malloc(matrix_mem_size);
    double *d_u = (double *)malloc(matrix_mem_size);

    HANDLE_ERROR(cudaMalloc((void **)&d_w, matrix_mem_size));
    HANDLE_ERROR(cudaMalloc((void **)&d_u, matrix_mem_size));

    HANDLE_ERROR(cudaMemcpy(d_w, w, matrix_mem_size, cudaMemcpyHostToDevice));

    dim3 dimGrid(16, 16);  // 256 blocks
    dim3 dimBlock(32, 32); // 1024 threads

    diff = epsilon;

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    cudaEventRecord(cudaStart, 0);

    while (epsilon <= diff)
    {
        copy_grid<<<dimGrid, dimBlock>>>(d_w, d_u);
        calculate_solution<<<dimGrid, dimBlock>>>(d_w, d_u);
        epsilon_reduction<<<256, 1024>>>(d_w, d_u);
        epsilon_reduction_results<<<256, 1024>>>();

        cudaDeviceSynchronize();

        HANDLE_ERROR(cudaMemcpyFromSymbol(&diff, d_epsilon, sizeof(double), 0, cudaMemcpyDeviceToHost));

        iterations++;
        if (iterations == iterations_print)
        {
            printf("  %8d  %lg\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }

    CHECK_CUDA_ERROR("Kernel invocation");

    cudaEventRecord(cudaStop, 0);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&ElapsedTime, cudaStart, cudaStop);

    printf("\n");
    printf("  %8d  %lg\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  GPU time = %f\n", ElapsedTime / 1000);

    HANDLE_ERROR(cudaMemcpy(w, d_w, matrix_mem_size, cudaMemcpyDeviceToHost));

    cudaFree(d_w);
    cudaFree(d_u);

    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
}

#undef M
#undef N
#undef NUM_ELEMENTS