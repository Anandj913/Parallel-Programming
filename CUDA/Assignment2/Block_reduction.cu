/* -----------------------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll No: 17EC30041
CUDA
Assignment 2: Block Reduction 
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce(float *d_A, float *d_B, int N, int K)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*(blockDim.x) + threadIdx.x;
	int i = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;
    int s;
    
	if(i<N)
	{
		for(s =1; s<K; s*= 2)
        {
           
            if(i %(2*s)==0 && i+s <N)
            {
                d_A[i] += d_A[i+s];
                __syncthreads();

            }
            
        }
        if(i % K == 0)
        {
            d_B[i/K]= d_A[i]/K;
        }
	}
}


int main(void)
{
	cudaError_t err = cudaSuccess;

	int p,q,T,N,K,call, j=1, final = 0; 
    float *d_A = NULL, *d_B = NULL, *h_B = NULL, *h_A = NULL;
    printf("\n Enter the number of test cases:");
    scanf("%d", &T);
    while(T>0)
    {
        int i;
        printf("\n Enter the number p:");
        scanf("%d", &p);
        printf("\n Enter the number q:");
        scanf("%d", &q);
        N = pow(2,p);
        K = pow(2,q);
        h_A = (float *)malloc(N*sizeof(float));
        printf("\n Enter elements of Array A:");
        for(i=0; i<N; i++)
        {
            scanf("%f", &h_A[i]);
        }
        call = p/q;
        printf("\n-------------| Running test case: %d |-------------", j);
        for(i=0;i<call;i++)
        {
            if(i != 0)
            {
                free(h_A);
                h_A = h_B;
            }
            cudaMalloc((void **)&d_A, N*sizeof(float));
            cudaMalloc((void **)&d_B, (N/K)*sizeof(float));
            h_B = (float *)malloc((N/K)*sizeof(float));
            err = cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "\nFailed to copy array from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            dim3 grid(sqrt(N/K),sqrt(N/K),1);
            dim3 block(K,1,1);
            printf("\nLaunching kernel for %d time", (i+1));
            reduce<<<grid,block>>>(d_A,d_B,N,K);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "\nFailed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            else
            {
                printf("\nkernel launched successfully");
            }
            err = cudaMemcpy(h_B, d_B, (N/K)*sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "\nFailed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            printf("\nReduced Array Size: %d\n", N/K);
            cudaFree(d_A);
            cudaFree(d_B);
            if(i == call-1)
            {
                final = N/K;
            }
            N = N/K;
        }

        printf("\nOutput Array for test case%d\n", j);
        printf("B[%d] = [", final);
        for(i=0; i< final; i++)
        {
            printf("%.2f ,", h_B[i]);
        }
        printf("]\n");
        free(h_A);
        free(h_B);
        j = j+1;
        T=T-1;
        printf("\n-------------| End of test case %d |-------------", j-1);
        
    }
	err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\nDone\n");
    return 0;
}