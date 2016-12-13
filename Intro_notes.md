GPU-Accelerated Applications
    - www.nvidia.com/appscatalog

3 Ways to Accelerate Applications

    - Libraries: "Drop-in" Acceleration

    - OpenACC Directives: Easily Accelerate Applications
        - Compiler Parallelizes code
        - Works on many-core GPUs & multicore CPUs

    - Programming Languages: Maximum Flexibility

Where to get help?
    - Registered Developer: developer.nvidia.com

Introduction to CUDA C/C++

Terminology
    - Host: The CPU and its memory (host memory)
    - Device: The GPU and its memory (device memory)

Simple Processing Flow

1. Copy input data from CPU memory to GPU memory
2. Load GPU program and execute, caching data on chip for performance
3. Copy results from GPU memory to CPU memory

Hello World! With Device code
     - __global__ indicates a function:
        - Runs on the device
        - Is called from the host code
Kernel:
    __global__ void mykernel(void) {
    }
Kernel Launch:
    mykernel<<<1,1>>>();


Moving to Parallel
    - add<<< N, 1 >>>();
    - Instead of executing add() once, execute N times in parallel
    - Terminology: each parallel invocation of add() is referred to as a block
        - The set of blocks is refereed to as a grid
        - Each invocation can refer to its block index using blockIdx.x

CUDA Threads
    - Terminology: a block can be split into parallel threads
    - threadIdx.x instead of blockIdx.x
    - thread limit of 1024 (choose whole numbers multiples of 32)

Combining Blocks and Threads
    - No longer as simple as using blockIdx.x and threadIdx.x
    - With M threads/block a unique index for each thread is given by:
        int index  = threadIdx.x + blockIdx.x * M;
    - Use built-n variable blockDim.x for threads per block
        int index  = threadIdx.x + blockIdx.x * blockDim.x;
    - Avoid accessing beyond the end of the arrays
    - Update the kernel launch:
        add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);

1D Stencil
    - If radius is 3, then each output element is the sum of the 7 input elements
    - Input elements are read several times
        - Width of 3, each input element is read seven times

Sharing Data between Threads
    - Terminology: within a block, threads share data via shared memory
    - Declare using __shared__, allocated per block
    - Data is not visible to threads in other blocks

Implementing With Shared Memory
    - Read (blockDim.x + 2 * radius) from global to shared memory

syncthreads()
    - synchronizes all threads within a block

Coordinating Host & Device
    - Kernel launches are asynchronous
    - CPU needs to synchronize before consuming the results
