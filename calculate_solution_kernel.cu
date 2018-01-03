#define M 50
#define N 50

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

// CUDA kernel to perform the reduction in parallel on the GPU
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
__global__ void calculate_solution(double *w, double *u, double epsilon, double diff)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > 0 && y > 0 && x < M - 1 && y < N - 1)
    {
        //printf("Hello from x %d and y %d and value %f\n", x, y, u[x * M + y]);
        //w[x * M + y] = (u[x * M + (y - 1)] + u[(x - 1) * M + y] + u[x * M + (y + 1)] + u[(x + 1) * M + y]) / 4.0;
        __syncthreads();
    }

    return;
}

void calculate_solution_kernel(double w[M][N], double epsilon, double diff)
{
    int i;
    int j;
    double u[M][N];

    //  Save the old solution in U.
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            u[i][j] = w[i][j];

    const unsigned int matrix_mem_size = sizeof(double) * M * N;
    //HANDLE_ERROR(cudaMalloc(&d_a, N*sizeof(char)));

    double *d_w = (double *)malloc(matrix_mem_size);
    double *d_u = (double *)malloc(matrix_mem_size);

    // Memory allocation on device side
    HANDLE_ERROR(cudaMalloc((void **)&d_w, matrix_mem_size));
    HANDLE_ERROR(cudaMalloc((void **)&d_u, matrix_mem_size));

    // Copy from host memory to device memory
    HANDLE_ERROR(cudaMemcpy(d_u, u, matrix_mem_size, cudaMemcpyHostToDevice));

    dim3 dimGrid(1, 1);
    dim3 dimBlock(M, N);

    // Invoke the kernel
    calculate_solution<<<1, 1024>>>(d_w, d_u, epsilon, diff);

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("kernel invocation");

    // Copy from device memory back to host memory
    HANDLE_ERROR(cudaMemcpy(w, d_w, matrix_mem_size, cudaMemcpyDeviceToHost));

    cudaFree(d_w);
    cudaFree(d_u);
}

#undef M
#undef N