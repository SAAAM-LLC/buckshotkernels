#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// Ternary matrix multiplication kernel optimized for {-1, 0, 1}
extern "C" __global__ void ternary_matmul_kernel(
    const int8_t* __restrict__ A,    // Input matrix A (M x K)
    const int8_t* __restrict__ B,    // Input matrix B (K x N)
    int32_t* __restrict__ C,         // Output matrix C (M x N) - int32 for accumulation
    int M, int N, int K
) {
    __shared__ int8_t As[16][16];
    __shared__ int8_t Bs[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t sum = 0;

    // Loop over tiles
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        int a_row = row;
        int a_col = tile * 16 + threadIdx.x;
        int b_row = tile * 16 + threadIdx.y;
        int b_col = col;

        // Load A tile
        As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0;
        // Load B tile
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0;

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            int8_t a_val = As[threadIdx.y][k];
            int8_t b_val = Bs[k][threadIdx.x];

            // Ternary optimization: only compute if both non-zero
            if (a_val != 0 && b_val != 0) {
                sum += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Ternary quantization: float -> {-1,0,1}
extern "C" __global__ void ternary_quantize_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float threshold,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx
