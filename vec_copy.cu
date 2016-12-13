#include <stdio.h>
#include <stdlib.h>

const int N = 2048;

__global__ void add(const float *a, float *c, int n){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < n)
        c[idx]=a[idx];
}



int main(){
    float *h_a, *h_c, *d_a, *d_c;
    const size_t ds = N*sizeof(float);
    h_a = (float *)malloc(ds);
    h_c = (float *)malloc(ds);
    cudaMalloc(&d_a, ds);
    cudaMalloc(&d_c, ds);
    for (int i = 0; i < N; i++){
        h_a[i] = rand()/(float)RAND_MAX;
        h_c[i] = 0;
    }

    cudaMemcpy(d_a, h_a, ds, cudaMemcpyHostToDevice);

    add<<<(N+255)/256, 256>>>(d_a, d_c, N);
    cudaMemcpy(h_c, d_c, ds, cudaMemcpyDeviceToHost);

    printf("h_a[0] = %f\n", h_a[0]);
    printf("h_c[0] = %f\n", h_c[0]);

    return 0;
}
