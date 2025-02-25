

# Implement MLP from Scratch with CUDA

>This is a lab project of my graduate level class.

## What it do

- Implemented an **MLP handwriting recognition classification model** from scratch using **C++** and **CUDA**, including tensor operations and network computation on the GPU.

- **Lab1**: completes all the operation header function for tensor

  - operator`op_elemwise.cuh`, `op_mm.cuh`, `op_reduction.cuh`. After filling them, we can pass some unit process, but can't realize this network. Note that matmul use tiling to accelerate.

- **Lab2**: Implementing Multilayer Perceptron (MLP). This include:

  - we construct a class to achieve this 2 layer MLP, forward and backward. we initialize its weights with `mlp.init()`. Instead of autograd, we performed a manual gradient calculation to get a better performance. 
  - realize `op_cross_entropy_loss` , which is a manually fused operator that calculates the loss given the logits tensor (computed by `model.forward`), the `targets` tensor containing the batch's training labels. Additionally, the operator also calculates the gradients of the logits and put them in the `d_logits` output tensor. With the `d_logits` gradients, we can start the rest of the backward computation by calling `model.backward(input_images, d_logits, d_input_images).`
  - Finally, we take a gradient descend step to update the model parameters using `sgd.step()`.

  The details are in `Lab1.md` and `Lab2.md`.

## Environment

I run it on school HPC. But I think any Linux Environment with `nvcc` environment and GPU will be OK.

```bash
which nvcc
nvidia-smi -L
```



## How It Run

### Lab 1 - test the ops

While you are writing the code and fixing the compilation errors, you do not need a GPU. Any CPU machine on the HPC cluster should be sufficient for compilation as they have `nvcc` (the CUDA compiler) installed.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```



Our lab uses the [cmake](https://cmake.org/cmake/help/latest/index.html) tool to generate a Makefile for the project. Once the Makefile is generated, we can use the `make` tool to compile our code.

Use make file and run it:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Test correctness

```python
$./test
slice passed...
op_add passed...
op_multiply passed...
op_sgd passed...
matmul passed...
op_sum passed...
op_cross_entropy_loss passed...
All tests completed successfully!
```

### Lab 2 - implete MLP

To train MLP in barenet, we need to first get the MNIST dataset. There are two ways to do it: One, you can run the script `mnist_mlp.ipnb` on your cloned lab repository on HPC, which will download and save the MNIST dataset. Alternatively, if you do not want to run `mnist_mlp.ipnb` on HPC, you can download the MNIST dataset using the following command

```
$ python download.py
```

You should see the subdirectory named `data/MNIST/raw` with all the MNIST training and test data files:

```
$ ls data/MNIST/raw
t10k-images-idx3-ubyte		t10k-labels-idx1-ubyte		train-images-idx3-ubyte		train-labels-idx1-ubyte
t10k-images-idx3-ubyte.gz	t10k-labels-idx1-ubyte.gz	train-images-idx3-ubyte.gz	train-labels-idx1-ubyte.gz
```

Next, compile the code. The compilation procedure is the same as that for Lab-1.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

After finishing compilation, do MLP training by typing the following:

```
$ cd build
$ ./train_mlp  
```

An example output from a working lab is shown below:

```
# of training datapoints=60000 # of test datapoints= 10000 feature size=784
training datapoints mean=0.131136
TRAINING epoch=0 loss=0.893307 accuracy=0.787633 num_batches=1875
TRAINING epoch=1 loss=0.390445 accuracy=0.891333 num_batches=1875
TRAINING epoch=2 loss=0.338284 accuracy=0.903617 num_batches=1875
TRAINING epoch=3 loss=0.313121 accuracy=0.910933 num_batches=1875
TRAINING epoch=4 loss=0.295669 accuracy=0.915917 num_batches=1875
TRAINING epoch=5 loss=0.282319 accuracy=0.919767 num_batches=1875
TRAINING epoch=6 loss=0.271276 accuracy=0.922933 num_batches=1875
TRAINING epoch=7 loss=0.261205 accuracy=0.924933 num_batches=1875
TRAINING epoch=8 loss=0.251797 accuracy=0.927883 num_batches=1875
TRAINING epoch=9 loss=0.243037 accuracy=0.93055 num_batches=1875
TEST epoch=0 loss=0.241526 accuracy=0.928986 num_batches=312
```



