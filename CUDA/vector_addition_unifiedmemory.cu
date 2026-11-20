
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void vector_add(float *A, float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int N = 8;
    size_t bytes = N * sizeof(float);

    float *A, *B, *C;

    // Allocate Unified Memory
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Initialize arrays using CPU
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
        B[i] = (i + 1) * 10;
    }

    // Get current GPU device
    int device;
    cudaGetDevice(&device);

    // Prefetch input and output arrays to GPU
    cudaMemPrefetchAsync(A, bytes, device);
    cudaMemPrefetchAsync(B, bytes, device);
    cudaMemPrefetchAsync(C, bytes, device);

    int threadperblock = 256;
    int block = (N + threadperblock - 1) / threadperblock;

    // Execute vector addition
    vector_add<<<block, threadperblock>>>(A, B, C, N);

    // Wait for GPU execution to finish
    cudaDeviceSynchronize();

    // Prefetch arrays back to CPU
    cudaMemPrefetchAsync(A, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(B, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(C, bytes, cudaCpuDeviceId);

    // Wait for prefetching to finish
    cudaDeviceSynchronize();

    // Print results
    cout << "Vector A: ";
    for (int i = 0; i < N; i++) {
        cout << A[i] << " ";
    }
    cout << endl;

    cout << "Vector B: ";
    for (int i = 0; i < N; i++) {
        cout << B[i] << " ";
    }
    cout << endl;

    cout << "Vector C (A + B): ";
    for (int i = 0; i < N; i++) {
        cout << C[i] << " ";
    }
    cout << endl;

    // Free Unified Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
//here d_A and h_A are the same pointer as they uses unified memory. The same applies to d_B and h_B, as well as d_C and h_C.