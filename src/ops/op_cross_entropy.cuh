#pragma once
#include "utils/tensor.cuh"

//This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
//and the batch's corresponding "target" label tensor and returns the average loss of the batch.
//It also returns the gradient of the logits tensor.

template <typename T>
__global__ void CrossEntropyGradientKernel(const Tensor<T> logits, const Tensor<char> targets, Tensor<T> d_logits) {

    int row = blockIdx.x * blockDim.x + threadIdx.x; // Batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Class index
    
    if (row < logits.h && col < logits.w) {
        int target_index = static_cast<int>(targets.rawp[targets.offset + row * targets.stride_h]);
        T logit_value = logits.rawp[logits.offset + row * logits.stride_h + col * logits.stride_w];
        // Compute gradient based on the condition
        //printf("Before change, logit_value[%d][%d] = %f\n", row, col, logit_value);
        if (col == target_index) {
            d_logits.rawp[d_logits.offset + row * d_logits.stride_h + col * d_logits.stride_w] = (logit_value - 1) / logits.h;
        } else {
            d_logits.rawp[d_logits.offset + row * d_logits.stride_h + col * d_logits.stride_w] = logit_value / logits.h;
        }
        //printf("After change, d_logits[%d][%d] = %f\n", row, col, d_logits.rawp[d_logits.offset + row * d_logits.stride_h + col * d_logits.stride_w]);
    }
}

template <typename T>
__global__ void CrossEntropyLossKernel(const Tensor<T> logits, const Tensor<char> targets, Tensor<T> loss_per_sample) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < logits.h) {
        int target_index = static_cast<int>(targets.rawp[targets.offset + idx * targets.stride_h]);
        T logit_value = logits.rawp[logits.offset + idx * logits.stride_h + target_index * logits.stride_w];
        // Assuming logits are already passed through log_softmax
        //printf("Before log, logit_value = %f\n", logit_value);
        loss_per_sample.rawp[idx] = -log(logit_value);  // Negative log probability for the correct class
        //printf("Loss per sample[%d] = %f\n", idx, loss_per_sample.rawp[idx]);
    }
}

template <typename T>
__global__ void compute_exp_kernel(Tensor<T> logits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < logits.h * logits.w) {
        //printf("Before exp, logits[%d] = %f\n", idx, logits.rawp[idx]);
        logits.rawp[idx] = exp(logits.rawp[idx]);
        //printf("After exp, logits[%d] = %f\n", idx, logits.rawp[idx]);
    }
}

template <typename T>
__global__ void apply_softmax_kernel(Tensor<T> logits, const Tensor<T> logits_sum) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < logits.h && col < logits.w) {
        T sum_value = logits_sum.rawp[logits_sum.offset + row * logits_sum.stride_h];
        int idx = logits.offset + row * logits.stride_h + col * logits.stride_w;
        //printf("Before softmax, logits[%d][%d] = %f, sum_value = %f\n", row, col, logits.rawp[idx], sum_value);
        logits.rawp[idx] = logits.rawp[idx] / sum_value;
        //printf("After softmax, logits[%d][%d] = %f\n", row, col, logits.rawp[idx]);
    }
}


template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> &targets,
                               Tensor<T> &d_logits)
{   
    Tensor<T> logits_tmp(logits.h, 1, true);
    assert(logits.h == targets.h && logits.h == d_logits.h && logits.h == logits_tmp.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1 && logits_tmp.w == 1);

    assert(logits.on_device && targets.on_device && d_logits.on_device && logits_tmp.on_device); 

    //Lab-2: please add your code here. 
    //You need to define separate GPU kernel function(s) and launch them here
    //In order to calculate d_logits, you should derive what its values should be 
    //symbolically.

    //Tensor<float> logits_host = logits.toHost(); // 将设备上的数据传回主机
    //Tensor<char> targets_host = targets.toHost();
    
    // 打印logits的尺寸
    //std::cout << "Logits size: " << logits_host.h << "x" << logits_host.w << std::endl;
    // 然后，使用Tensor类的str()方法打印logits的内容
    //std::cout << "Logits: " << std::endl << logits_host.str() << std::endl;

    // 打印targets的尺寸
    //std::cout << "Targets size: " << targets_host.h << "x" << targets_host.w << std::endl;

    
    // compute logit p
    dim3 block(256);
    dim3 grid((logits.h * logits.w + block.x - 1) / block.x);
    
    compute_exp_kernel<<<grid, block>>>(logits);
    op_sum(logits, logits_tmp);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((logits.h + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (logits.w + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    apply_softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(logits, logits_tmp);

    // compute gradient
    CrossEntropyGradientKernel<<<blocksPerGrid, threadsPerBlock>>>(logits, targets, d_logits);

    // compute loss (reuse space)
    CrossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(logits, targets, logits_tmp);
    Tensor<T> loss_sum{1, 1};
    Tensor<T> logits_tmp_host = logits_tmp.toHost();
    T mean_loss = logits_tmp_host.mean();

    cudaAssert(cudaPeekAtLastError());
    cudaAssert(cudaDeviceSynchronize());

    //std::cout << "Mean loss: " << mean_loss << std::endl;
    return mean_loss;

}
