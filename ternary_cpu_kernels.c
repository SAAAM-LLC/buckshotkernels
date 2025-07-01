#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __x86_64__
#include <immintrin.h>  // For AVX/SSE intrinsics
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>   // For ARM NEON intrinsics
#endif

// =========================
// Ternary Matrix Multiply
// =========================

void ternary_matmul_cpu(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    // Clear output
    memset(C, 0, M * N * sizeof(int32_t));
    const int BLOCK_SIZE = 64;

    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {

                int i_max = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
                int j_max = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                int k_max = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        int32_t sum = C[i * N + j];
                        int k = k0;

#ifdef __AVX2__
                        // AVX2 vectorized computation
                        __m256i acc = _mm256_setzero_si256();
                        for (; k <= k_max - 32; k += 32) {
                            __m256i a_vec = _mm256_loadu_si256((__m256i*)(A + i * K + k));
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)(B + k * N + j));
                            __m256i prod = _mm256_maddubs_epi16(a_vec, b_vec);
                            acc = _mm256_add_epi16(acc, prod);
                        }
                        __m128i acc_low = _mm256_extracti128_si256(acc, 0);
                        __m128i acc_high = _mm256_extracti128_si256(acc, 1);
                        acc_low = _mm_add_epi16(acc_low, acc_high);
                        int16_t temp[8];
                        _mm_storeu_si128((__m128i*)temp, acc_low);
                        for (int t = 0; t < 8; ++t) {
                            sum += temp[t];
                        }
#endif
                        // Scalar fallback
                        for (; k < k_max; ++k) {
                            int8_t a_val = A[i * K + k];
                            int8_t b_val = B[k * N + j];
                            if (a_val != 0 && b_val != 0) {
                                sum += a_val * b_val;
                            }
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// =========================
// Ternary 2D Convolution
// =========================

void ternary_conv2d_cpu(
    const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int N, int C, int H, int W,
    int Out_C, int KH, int KW,
    int Out_H, int Out_W,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    memset(output, 0, N * Out_C * Out_H * Out_W * sizeof(int32_t));
