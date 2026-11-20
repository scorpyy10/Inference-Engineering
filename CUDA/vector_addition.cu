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

    float A[8] = {1,2,3,4,5,6,7,8};
    float B[8] = {10,20,30,40,50,60,70,80};
    float C[8];
    // Initialize device pointers to store address of device memory
    float *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    //copy data from host to device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    int threadperblock = 256;
    // Calculate number of blocks needed to cover all elements
    int block = (N + threadperblock - 1) / threadperblock;

    vector_add<<<block, threadperblock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}