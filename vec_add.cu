#include <stdio.h>
#include <stdlib.h>
#define DSIZE 256
#define nTPB 64

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

__global__ void vector_add_kernel(float *c, const float *a, const float *b){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < DSIZE)
    c[idx] = a[idx] + b[idx];
}

int main(){
  float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
  h_a = (float *)malloc(DSIZE * sizeof(float));
  h_b = (float *)malloc(DSIZE * sizeof(float));
  h_c = (float *)malloc(DSIZE * sizeof(float));
  if (h_a == NULL) {printf("malloc fail\n"); return 1;}
  if (h_b == NULL) {printf("malloc fail\n"); return 1;}
  if (h_c == NULL) {printf("malloc fail\n"); return 1;}
  for (int i = 0; i < DSIZE; i++){
    h_c[i] = 0.0f;
    h_a[i] = rand()/(float)RAND_MAX;
    h_b[i] = rand()/(float)RAND_MAX;}

  cudaMalloc(&d_a, DSIZE*sizeof(float));
  cudaMalloc(&d_b, DSIZE*sizeof(float));
  cudaMalloc(&d_c, DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc fail");
  cudaMemcpy(d_a, h_a, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_c, 0, DSIZE*sizeof(float));
  cudaCheckErrors("cudaMemcpy H2D fail");
  dim3 threads(nTPB, 1, 1);
  dim3 blocks((DSIZE+threads.x-1)/threads.x, 1, 1);
  vector_add_kernel<<<blocks, threads>>>(d_c, d_a, d_b);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel fail");
  cudaMemcpy(h_c, d_c, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H fail");
  printf("h_a[0] = %f\n", h_a[0]);
  printf("h_b[0] = %f\n", h_b[0]);
  printf("h_c[0] = %f\n", h_c[0]);
  return 0;
}
