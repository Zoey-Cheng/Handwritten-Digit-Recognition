# Lab-1: Implementing basic tensor operators on GPUs.

The goal of this lab is two-fold. One, we want to demonstrate the abstraction of tensors and tensor operations.
Two, we want you to become familiar with GPU programming so as to implement the tensor operations.

## Preliminary: Obtaining the lab code
Follow the instruction given on Campuswire to create your github repository containing the lab skeleton files.
Then, on a HPC machine, type this:
```
git clone git@github.com:nyu-mlsys-sp24/barenet-<YourGithubUsername>.git barenet
```

### Compilation
While you are writing the code and fixing the compilation errors, you do not need a GPU.  Any CPU machine on the HPC cluster 
should be sufficient for compilation as they have `nvcc` (the CUDA compiler) installed. 

To compile, do the following in the lab's directory `barenet`:
```
$ cd barenet
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Our lab uses the [cmake](https://cmake.org/cmake/help/latest/index.html) tool
to generate a Makefile for the project. Once the Makefile is generated, we can use
the `make` tool to compile our code.

## Correctness
Correctness will constitute 70% of Lab-1's score. 
We will evaluate the correctness of your Lab-1 using a simple unit test file `test.cu`.  You need a GPU in order to run the test! 
Once you finish compilation, 
run the unit test as follows:

```
$ ./test
```

If your passed the unit test, the output will look like this:
```
$./test
slice passed...
op_add passed...
op_multiply passed...
op_sgd passed...
matmul passed...
op_sum passed...
All tests completed successfully!
```

## Performance
Performance will constitute 30% of Lab-1's score. We will compare your kernel's performance with those of the instructor's 
own basic implementation.

## Lab-1 instructions

In Lab-1, you will complete the necessary code in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh` and pass 
simple unit tests. Your implementation should also be performant in order to get good performance scores.

You should first carefully read through the code in `utils/tensor.cuh`. This file defines our tensor abstraction. 
Here are the things you should know about our tensors:

- The tensor is templated so that we can have different data types for the elements stored in it.

- The tensor's internal buffer is ref-counted using C++ smart pointer and its corresponding memory storage is automatically freed when there are no more references to it.
In other words, you do not need to worry about needing to free the memory.

- For our labs, the tensor is *always* 2-dimensional, with the first dimension named `h` (height) and second dimension named `w` (width).
A row vector has `h=1` and a column vector has `w=1`.

- The Macro `Index(t, i,j)` will  be handy for accessing the element at [i,j] coordinate of tensor t. The Macro `IndexOutOfBound(t, i, j)` will be handy for testing whether [i,j] is out of bound for tensor t.


Next, complete the necessary functions  in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh`, in the given order.
Read through the code skeleton, and fill in your code whenever you see a comment that says `Lab-1: please add your code here`.
After finishing each file, you should be able to pass a portion of the unit test. Debug and complete that portion of the unit test 
before moving on to the next lab file.


