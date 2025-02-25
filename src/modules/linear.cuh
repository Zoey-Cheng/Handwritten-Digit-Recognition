#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"

template<typename T>
class LinearLayer {
    private:
        int in_dim;
        int out_dim;

        Parameter<T> w;
        Parameter<T> b;

    public:
    LinearLayer(int in_dim_, int out_dim_, bool gpu):in_dim(in_dim_), out_dim(out_dim_) {
        w = Parameter<T>{in_dim, out_dim, gpu};
        b = Parameter<T>{1, out_dim, gpu};
    }

    LinearLayer() {}
    
    LinearLayer(LinearLayer&& other) : in_dim(other.in_dim), out_dim(other.out_dim), w(other.w), b(other.b) {}
                
    std::vector<Parameter<T>*> parameters() {
        std::vector<Parameter<T> *> v;
        v.push_back(&w);
        v.push_back(&b);
        return v;
    }
    
    void init_uniform() {
        // Do Kaiming uniform
        float max = 1.0f / std::sqrt(in_dim);
        //std::cout << "May be error here..." << std::endl;
        op_uniform_init(w.t, -max, max);
        //std::cout << "Pass op_uniform_init 1..." << std::endl;
        op_uniform_init(b.t, -max, max);
        //std::cout << "Pass op_uniform_init 2..." << std::endl;
        //std::cout << "init b=" << b.t.str() << std::endl;
    }

    //This function calculates the output of a lienar layer 
    //and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
        //Lab-2: please add your code here
        
        // y = x * w + b
        // Matrix multiply input 'x' with weights 'w' to compute 'y'
        //std::cout << "Problem is here..." << std::endl;
        //std::cout << "x: " << x.h << " * " << x.w << std::endl;
        //std::cout << "w.t: " << w.t.h << " * " << w.t.w << std::endl;
        op_mm(x, w.t, y); 
        //std::cout << "Problem Pass 1..." << std::endl;
        // Add bias 'b' to each row of the matrix 'y'
        op_add(y, b.t, y);
        //std::cout << "Problem Pass 2..." << std::endl;

    }

    //This function performs the backward operation of a linear layer
    //Suppose y = Linear(x). Then function argument "dy" is the gradients of "y", 
    //and function argument "x" is the saved x.
    //This function compute the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    //It also computes the graidents of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
        //Lab-2: Please add your code here

        // Compute gradient with respect to weights (dw) by matrix multiplication of x's transpose and dy
        Tensor<float> xt = x.transpose();
        //std::cout << "Problem is backward 1..." << std::endl;
        //std::cout << "xt: " << xt.h << " * " << xt.w << std::endl;
        //std::cout << "w.t: " << w.t.h << " * " << w.t.w << std::endl;
        op_mm(xt, dy, w.dt); // w.dt = dY/dw_i

        // Compute gradient with respect to bias (db) as the sum of dy across all samples in the batch
        op_sum(dy, b.dt); // b.dt = sum (dY/dw_i), in batch_size direction

        // Compute gradient with respect to input (dx) by matrix multiplication of dy and weights' transpose
        Tensor<float> wt = w.t.transpose();
        op_mm(dy, wt, dx);

        //std::cout << "Raw dy: " << std::endl;
        //auto dy_host = dy.toHost();
        //std::cout << dy_host.str() << std::endl;

        //std::cout << "bias gradient: " << std::endl;
        //auto b_dt_host = b.dt.toHost();
        //std::cout << b_dt_host.str() << std::endl;

        //std::cout << "Final Output dx: " << std::endl;
        //auto dx_host = dx.toHost();
        //std::cout << dx_host.str() << std::endl;
        
    }

};
