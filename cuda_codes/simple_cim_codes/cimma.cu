#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <stdint.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__global__ void mma16816NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                    size_t M, size_t N, size_t K) {
    const size_t K_tiles = DIV_CEIL(K, MMA_K);

    //划分到当前线程块的输出矩阵块的起始行和列。
    const size_t warpRow = blockIdx.x * MMA_M;
    const size_t warpCol = blockIdx.y * MMA_N;

    if (warpRow >= M || warpCol >= N) {
        return;
    }
    //shmem_A: 16*16
    __shared__ half shmem_A[MMA_M][MMA_K];
    //shmem_B: 8*16
    __shared__ half shmem_B[MMA_N][MMA_K];
    //shmem_C: 16*8
    __shared__ half shmem_C[MMA_M][MMA_N];

    const size_t laneId = threadIdx.x % WARP_SIZE;

    // uint32_t RA[4];
    // uint32_t RB[2];
    // uint32_t RC[2] = {0, 0};

    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&shmem_A[laneId / 2][0]) + laneId % 2) =
            *((int4 *)(&A[(warpRow + laneId / 2) * K + i * MMA_K]) + laneId % 2);

        if (laneId < MMA_N * 2) {
            *((int4 *)(&shmem_B[laneId / 2][0]) + laneId % 2) =
                *((int4 *)(&B[i * MMA_K + (warpCol + laneId / 2) * K]) + laneId % 2);
        }

        __syncthreads();

        // uint32_t shmem_A_lane_addr = __cvta_generic_to_shared(&shmem_A[laneId % 16][(laneId / 16) * 8]);
        // asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        //              : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
        //              : "r"(shmem_A_lane_addr));

        // uint32_t shmem_B_lane_addr = __cvta_generic_to_shared(&shmem_B[laneId % 8][((laneId / 8) % 2) * 8]);
        // asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        //              : "=r"(RB[0]), "=r"(RB[1])
        //              : "r"(shmem_B_lane_addr));

        // asm volatile(
        //     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
        //     : "=r"(RC[0]), "=r"(RC[1])
        //     : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));

        // *((uint32_t *)(&shmem_C[laneId / 4][0]) + laneId % 4)     = RC[0];
        // *((uint32_t *)(&shmem_C[laneId / 4 + 8][0]) + laneId % 4) = RC[1];

        uint64_t shmem_C_addr = __cvta_generic_to_shared(&shmem_C[0][0]);

        // asm volatile ("cimma.shmma.synchro.rowmajor.colmajor.m16n8k16.f16.f16 [%0], [%1], [%2];" : "=l"(shmem_C) : "l"(shmem_A), "l"(shmem_B) : "memory");
        //对于功能模拟，当更换模拟CTA时，必须将shmem_C全部置零，否则因为功能模拟所有CTA用到的都是同一块shmem_C，会导致结果出现累加导致错误。
        asm volatile ("cimma.shmma.synchro.rowmajor.colmajor.m16n8k16.f16.f16 [%0], [%1], [%2];" : : "l"(shmem_C), "l"(shmem_A), "l"(shmem_B) : "memory");

        __syncthreads();
    }

    if (laneId < MMA_M) {
        *((int4 *)(&C[(warpRow + laneId) * N + warpCol])) = *((int4 *)(&shmem_C[laneId][0]));
    }

    __syncthreads();
}

void hgemmMmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    //M×N的输出矩阵被划分为MMA_M×MMA_N的小矩阵块，每个线程块(32个线程)计算一个小矩阵块。每个小矩阵块的大小是MMA_M×MMA_N。
    dim3 grid(DIV_CEIL(M, MMA_M), DIV_CEIL(N, MMA_N));

    mma16816NaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%MMA_M == 0); 
    assert( N%MMA_N == 0); 
    assert( K%MMA_K == 0); 

    size_t bytes_A = sizeof(half) * M * K;
    size_t bytes_B = sizeof(half) * K * N;
    size_t bytes_C = sizeof(half) * M * N;
    half* h_A = (half*)malloc(bytes_A);
    half* h_B = (half*)malloc(bytes_B);
    half* h_C = (half*)malloc(bytes_C);

    size_t bytes_A_cublas = sizeof(float) * M * K;
    size_t bytes_B_cublas = sizeof(float) * K * N;
    size_t bytes_C_cublas = sizeof(float) * M * N;
    float* h_A_cublas = (float*)malloc(bytes_A_cublas);
    float* h_B_cublas = (float*)malloc(bytes_B_cublas);
    float* h_C_cublas = (float*)malloc(bytes_C_cublas);

    half* d_A;
    half* d_B;
    half* d_C;

    // float* d_A_cublas;
    // float* d_B_cublas;
    // float* d_C_cublas;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    // checkCudaErrors(cudaMalloc(&d_A_cublas, bytes_A_cublas));
    // checkCudaErrors(cudaMalloc(&d_B_cublas, bytes_B_cublas));
    // checkCudaErrors(cudaMalloc(&d_C_cublas, bytes_C_cublas));

    // generate Matrix A's data
    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = __float2half((i%13) / 13.);
        h_A_cublas[i] = float((i%13) / 13.);
    }

    // generate Matrix B's data
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = __float2half((i%14) / 14.);
        h_B_cublas[i] = float((i%14) / 14.);
    }

    // generate Matrix C's data
    for( int i = 0; i < M * N; i++ ) {
        h_C[i] = __float2half(0.0);
        h_C_cublas[i] = float(0.0);
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy( d_A_cublas, h_A_cublas, bytes_A_cublas, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy( d_B_cublas, h_B_cublas, bytes_B_cublas, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy( d_C_cublas, h_C_cublas, bytes_C_cublas, cudaMemcpyHostToDevice));

    hgemmMmaNaive(d_A, d_B, d_C, M, N, K);

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // cublas
    // cublasHandle_t blas_handle;  
    // cublasCreate(&blas_handle);
    // half alpha = 1.0;
    // half beta = 0;
    
    // cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
    //     M, N, K, &alpha, 
    //     d_A_cublas, K, d_B_cublas, N, &beta, d_C_cublas, N
    // );

    // checkCudaErrors(cudaMemcpy( h_C_cublas, d_C_cublas, bytes_C_cublas, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                h_C_cublas[i * N + j] += __half2float(h_A_cublas[i * K + k] * 
                                                      h_B_cublas[j * K + k]);
                // printf("@@@ ### i-j-k: %d-%d-%d %f * %f => %f\n", i+1, j+1, k+1, 
                //        (h_A_cublas[i * K + k]), (h_B_cublas[j * K + k]), h_C_cublas[i * N + j]);
            }
            
        }
    }

    double eps = 1e-2;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs((float)h_C[i] - h_C_cublas[row * N + col]);
        double dot_length = 1.;
        double abs_val = fabs((float)h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%d][%d]=%.8f, ref=%.8f error term is > %E\n",
                    row, col, (float)h_C[i], h_C_cublas[row * N + col], eps);
            correct = false;
            // break;
        } else {
            if (i % (int)(M*N/10) == 0)
                printf("@@@ Right! Matrix[%d][%d]=%.8f, ref=%.8f\n", row, col, (float)h_C[i], h_C_cublas[row * N + col]);
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // cudaFree(d_A_cublas);
    // cudaFree(d_B_cublas);
    // cudaFree(d_C_cublas);
    
    free(h_A);
    free(h_B);
    free(h_C);

    free(h_A_cublas);
    free(h_B_cublas);
    free(h_C_cublas);
    
    return 0;
}



/* 
// OLD CODE
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 
#include <stdint.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__global__ void mma16816NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                    size_t M, size_t N, size_t K) {
    const size_t K_tiles = DIV_CEIL(K, MMA_K);

    const size_t warpRow = blockIdx.x * MMA_M;
    const size_t warpCol = blockIdx.y * MMA_N;

    if (warpRow >= M || warpCol >= N) {
        return;
    }
    //shmem_A: 16*16
    __shared__ half shmem_A[MMA_M][MMA_K];
    //shmem_B: 8*16
    __shared__ half shmem_B[MMA_N][MMA_K];
    //shmem_C: 16*8
    __shared__ half shmem_C[MMA_M][MMA_N];

    const size_t laneId = threadIdx.x % WARP_SIZE;

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};

    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&shmem_A[laneId / 2][0]) + laneId % 2) =
            *((int4 *)(&A[(warpRow + laneId / 2) * K + i * MMA_K]) + laneId % 2);

        if (laneId < MMA_N * 2) {
            *((int4 *)(&shmem_B[laneId / 2][0]) + laneId % 2) =
                *((int4 *)(&B[i * MMA_K + (warpCol + laneId / 2) * K]) + laneId % 2);
        }

        __syncthreads();

        uint32_t shmem_A_lane_addr = __cvta_generic_to_shared(&shmem_A[laneId % 16][(laneId / 16) * 8]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
                     : "r"(shmem_A_lane_addr));

        uint32_t shmem_B_lane_addr = __cvta_generic_to_shared(&shmem_B[laneId % 8][((laneId / 8) % 2) * 8]);
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(RB[0]), "=r"(RB[1])
                     : "r"(shmem_B_lane_addr));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
            : "=r"(RC[0]), "=r"(RC[1])
            : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));

        *((uint32_t *)(&shmem_C[laneId / 4][0]) + laneId % 4)     = RC[0];
        *((uint32_t *)(&shmem_C[laneId / 4 + 8][0]) + laneId % 4) = RC[1];

        __syncthreads();
    }

    if (laneId < MMA_M) {
        *((int4 *)(&C[(warpRow + laneId) * N + warpCol])) = *((int4 *)(&shmem_C[laneId][0]));
    }

    __syncthreads();
}

void hgemmMmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(DIV_CEIL(M, MMA_M), DIV_CEIL(N, MMA_N));

    mma16816NaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%MMA_M == 0); 
    assert( N%MMA_N == 0); 
    assert( K%MMA_K == 0); 

    size_t bytes_A = sizeof(half) * M * K;
    size_t bytes_B = sizeof(half) * K * N;
    size_t bytes_C = sizeof(half) * M * N;
    half* h_A = (half*)malloc(bytes_A);
    half* h_B = (half*)malloc(bytes_B);
    half* h_C = (half*)malloc(bytes_C);

    size_t bytes_A_cublas = sizeof(float) * M * K;
    size_t bytes_B_cublas = sizeof(float) * K * N;
    size_t bytes_C_cublas = sizeof(float) * M * N;
    float* h_A_cublas = (float*)malloc(bytes_A_cublas);
    float* h_B_cublas = (float*)malloc(bytes_B_cublas);
    float* h_C_cublas = (float*)malloc(bytes_C_cublas);

    half* d_A;
    half* d_B;
    half* d_C;

    float* d_A_cublas;
    float* d_B_cublas;
    float* d_C_cublas;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    checkCudaErrors(cudaMalloc(&d_A_cublas, bytes_A_cublas));
    checkCudaErrors(cudaMalloc(&d_B_cublas, bytes_B_cublas));
    checkCudaErrors(cudaMalloc(&d_C_cublas, bytes_C_cublas));

    // generate Matrix A's data
    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = (half)(i / 13);
        h_A_cublas[i] = (float)(i / 13);
    }

    // generate Matrix B's data
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = (half)(i / 14);
        h_B_cublas[i] = (float)(i / 14);
    }

    // generate Matrix C's data
    for( int i = 0; i < M * N; i++ ) {
        h_C[i] = (half)0.0;
        h_C_cublas[i] = (float)0.0;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy( d_A_cublas, h_A_cublas, bytes_A_cublas, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B_cublas, h_B_cublas, bytes_B_cublas, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_C_cublas, h_C_cublas, bytes_C_cublas, cudaMemcpyHostToDevice));

    hgemmMmaNaive(d_A, d_B, d_C, M, N, K);

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    
    cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
        M, N, K, &alpha, 
        d_A_cublas, K, d_B_cublas, N, &beta, d_C_cublas, N
    );

    checkCudaErrors(cudaMemcpy( h_C_cublas, d_C_cublas, bytes_C_cublas, cudaMemcpyDeviceToHost));

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs((float)h_C[i] - h_C_cublas[col * M + row]);
        double dot_length = M;
        double abs_val = fabs((float)h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%d][%d]=%.8f, ref=%.8f error term is > %E\n",
                    row, col, (float)h_C[i], h_C_cublas[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFree(d_A_cublas);
    cudaFree(d_B_cublas);
    cudaFree(d_C_cublas);
    
    free(h_A);
    free(h_B);
    free(h_C);

    free(h_A_cublas);
    free(h_B_cublas);
    free(h_C_cublas);

    return 0;
}
*/