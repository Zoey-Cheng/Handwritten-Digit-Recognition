# Training MNIST from Scratch with CUDA

>This is a lab project of my graduate level class.

## What it do

- Implemented an **MLP handwriting recognition classification model** from scratch using **C++** and **CUDA**, including tensor operations and network computation on the GPU.
- **Lab1** completes all the operation header function for tensor: `op_elemwise.cuh`, `op_mm.cuh`, `op_reduction.cuh`. After filling them, we can pass some unit process, but can't realize this network
-  **Lab2** completes detailed implementation of network, mainly `linear.cuh` and `mlp.cuh`
- **Accuracy**: Eventually it should be possible to obtain a multilayer perceptron model (MLP) with about the same accuracy as the pytorch version: between 97%-99+%. The final accuracy on test set is 98.26%.
- **Plus**: I'm wondering how much speed it can improve compared to the PyTorch version. I will try this part.

## Dataset - MINST

**Context**

MNIST is a subset of a larger set available from NIST (it's copied from http://yann.lecun.com/exdb/mnist/)

**Content**

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. .

Four files are available [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset):

- train-images-idx3-ubyte.gz: training set images
- train-labels-idx1-ubyte.gz: training set labels
- t10k-images-idx3-ubyte.gz: test set images
- t10k-labels-idx1-ubyte.gz: test set labels

## Environment

I run it on school HPC. But I think any Linux Environment with `nvcc` environment and GPU will be OK.

```bash
which nvcc
nvidia-smi -L
```

## How to run

Use make file and run it:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./test
```

