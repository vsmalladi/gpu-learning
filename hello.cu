#include <stdio.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void mykernel(){
  printf("Hello from block %d, thread %d.\n", blockIdx.x, threadIdx.x);
}

int main(){

  mykernel<<<2,2>>>();
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel fail");
  return 0;
}
