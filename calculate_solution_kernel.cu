#define M 64
#define N 64
#define NUM_ELEMENTS 4096

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

    if (x >= 0 && y >= 0 && x < M && y < N)
    {
        int index = x + y * N;

        d_u[index] = d_w[index];
        __syncthreads();
    }

    return;
}

__device__ double d_epsilon;

__device__ double epsilon_reduction_max[NUM_ELEMENTS];

__global__ void epsilon_reduction(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= 0 && y >= 0 && x < M && y < N)
    {
        int index = x + y * N;

        epsilon_reduction_max[index] = fabs(d_w[index] - d_u[index]);
        __syncthreads();

        for (unsigned int stride = NUM_ELEMENTS / 2; stride > 0; stride >>= 1)
        {
            if (index < stride)
                epsilon_reduction_max[index] = max(epsilon_reduction_max[index], epsilon_reduction_max[index + stride]); 
            __syncthreads();
        }
    }

    if (x == 0 && y == 0)
    {
        d_epsilon = epsilon_reduction_max[x + y * N];
    }

    return;
}

// CUDA kernel to perform the reduction in parallel on the GPU
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
__global__ void calculate_solution(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > 0 && y > 0 && x < M - 1 && y < N - 1)
    {
        int index = x + y * N;

        int left = (x - 1) + y * N;
        int right = (x + 1) + y * N;
        int top = x + (y - 1) * N;
        int bottom = x + (y + 1) * N;

        d_w[index] = (d_u[left] + d_u[right] + d_u[top] + d_u[bottom]) / 4.0;
        __syncthreads();
    }

    return;
}

void calculate_solution_kernel(double w[M][N], double epsilon)
{
    double diff;
    int iterations;
    int iterations_print;

    const unsigned int matrix_mem_size = sizeof(double) * M * N;

    double *d_w = (double *)malloc(matrix_mem_size);
    double *d_u = (double *)malloc(matrix_mem_size);

    // Memory allocation on device side
    HANDLE_ERROR(cudaMalloc((void **)&d_w, matrix_mem_size));
    HANDLE_ERROR(cudaMalloc((void **)&d_u, matrix_mem_size));

    // Copy from host memory to device memory
    HANDLE_ERROR(cudaMemcpy(d_w, w, matrix_mem_size, cudaMemcpyHostToDevice));

    // Dimensions for a 2D matrix with max size 512
    dim3 dimGrid(16, 16);  // 256 blocks
    dim3 dimBlock(32, 32); // 1024 threads

    diff = epsilon;

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    while (epsilon <= diff)
    {
        copy_grid<<<dimGrid, dimBlock>>>(d_w, d_u);
        calculate_solution<<<dimGrid, dimBlock>>>(d_w, d_u);
        epsilon_reduction<<<dimGrid, dimBlock>>>(d_w, d_u);

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

    printf("\n");
    printf("  %8d  %lg\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    //printf("  CPU time = %f\n", ctime);

    // Copy from device memory back to host memory
    HANDLE_ERROR(cudaMemcpy(w, d_w, matrix_mem_size, cudaMemcpyDeviceToHost));

    cudaFree(d_w);
    cudaFree(d_u);
}

#undef M
#undef N