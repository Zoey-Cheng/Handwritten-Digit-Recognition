#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

//This operator compute C = A@B
template <typename T>
__global__ void MatrixMultiKernal(const Tensor<T> A, const Tensor<T> B, Tensor<T> C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.h && col < C.w) {
        T sum = 0;
        for (int k = 0; k < A.w; k++) {
            sum += A.rawp[A.offset + row * A.stride_h + k * A.stride_w] * 
                   B.rawp[B.offset + k * B.stride_h + col * B.stride_w];
        }
        C.rawp[C.offset + row * C.stride_h + col * C.stride_w] = sum;
    }
}

template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    //std::cout << "A = " << A.h << "*" << A.w << std::endl;
    //std::cout << "B = " << B.h * B.w << std::endl;
    //std::cout << "C = " << C.h * C.w << std::endl;
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    //Lab-1: please complete this
    //You need to define separate kernel function(s) and launch them here
    //delete assert(0) when you are finished
    //assert(0);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((C.w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (C.h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMultiKernal<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);
    cudaAssert(cudaPeekAtLastError());
    cudaAssert(cudaDeviceSynchronize());
}
