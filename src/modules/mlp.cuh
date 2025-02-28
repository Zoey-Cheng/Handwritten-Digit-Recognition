#pragma once
#include "modules/linear.cuh"
template <typename T>
class MLP
{
private:
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> d_activ;

    int batch_size;
    int in_dim;

public:
    MLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_)
    {
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                layers.emplace_back(in_dim, layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        d_activ.reserve(layer_dims.size() - 1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {   
            //std::cout << "activ[" << i << "] has size" << batch_size << " * " <<layer_dims[i]<<std::endl;
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            // technically, i do not need to save d_activ for backprop, but since iterative
            // training does repeated backprops, reserving space for these tensor once is a good idea
            d_activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    std::vector<Parameter<T> *> parameters()
    {
        std::vector<Parameter<T> *> params;
        for (int i = 0; i < layer_dims.size(); i++)
        {
            auto y = layers[i].parameters();
            params.insert(params.end(), y.begin(), y.end());
        }
        return params;
    }

    void init() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].init_uniform();
        }
    }

    //This function peforms the forward operation of a MLP model
    //Specifically, it should call the forward oepration of each linear layer 
    //Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out)
    {
        //Lab-2: add your code here
        Tensor<T> current = in;
        for (int i = 0; i < layers.size() - 1; i++){
            //std::cout << "Before layer " << i << " forward call:" << std::endl;
            //std::cout << "current dimensions: (" << current.h << ", " << current.w << ")" << std::endl;
            //std::cout << "activ[" << i << "] dimensions: (" << activ[i].h << ", " << activ[i].w << ")" << std::endl;
            
            layers[i].forward(current, activ[i]);
            op_relu(activ[i], activ[i]);
            current = activ[i];
        }
        layers[layers.size()-1].forward(current, out);
    }

    //This function perofmrs the backward operation of a MLP model.
    //Tensor "in" is the gradients for the outputs of the last linear layer (aka d_logits from op_cross_entropy_loss)
    //Invoke the backward function of each linear layer and Relu from the last one to the first one.
    void backward(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
    {
        //Lab-2: add your code here
        Tensor<T> current_grad = d_out;
        for (int i = layers.size() - 1; i > 0; i--) {
            layers[i].backward(activ[i - 1], current_grad, d_activ[i - 1]);
            // Apply gradient of ReLU for all but the last layer
            op_relu_back(activ[i - 1], d_activ[i - 1], d_activ[i - 1]);
            current_grad = d_activ[i - 1];
        }
        // last layer
        layers[0].backward(in, current_grad, d_in);
    }
};
