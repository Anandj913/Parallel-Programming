/* -----------------------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll No: 17EC30041
CUDA
Assignment 1: 2D convolution operation for a 2D Matrix of floating point numbers. 
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ void RUN(cudaError_t call)
{
	cudaError_t err = call;
	if(err != cudaSuccess)
	{
		fprintf(stderr, " Failed with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__device__ float value(int i, int j, float *data, int N)
{
	if(i<0 || i>=N)
	{
		return 0;
	}
	else if(j<0 || j>=N)
	{
		return 0;
	}
	else
	{
		return data[i*N + j];
	}
}

__global__ void conv_2D(float *in, float *out, int N)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<N && j<N)
	{
		out[i*N + j] = (1.0/9.0)*(value(i-1,j-1,in,N) + value(i-1, j, in, N) + value(i-1, j+1, in, N) + value(i,j-1,in,N) + value(i,j,in,N) + value(i,j+1,in,N) + value(i+1,j-1,in,N) + value(i+1,j,in,N) + value(i+1,j+1,in,N));
	}
}
int main()
{
	cudaEvent_t startEvent, stopEvent;
	int max_x = 32; //maximum_thread_per_block 1024
	int test_case, N, k=1;
	long int i,j; 
	float *d_A, *h_A, *d_B, *h_B, ms;
	printf("\n Enter the number of test cases:");
	scanf("%d", &test_case);
	printf(" %d\n", test_case);

	RUN(cudaSetDevice(0));
	while(test_case)
	{
		RUN(cudaEventCreate(&startEvent));
		RUN(cudaEventCreate(&stopEvent));
		printf("\nRunning test case: %d",k);
		printf("\n Enter dimention of Matrix:");
		scanf("%d", &N);
		printf(" %d\n", N);
		h_A = (float *)malloc(N*N*sizeof(float));
    	h_B = (float *)malloc(N*N*sizeof(float));
		printf("\n Enter entries of input matrix:\n");
		for(i=0; i<N*N; i++)
		{
			scanf("%f", &h_A[i]);
		}
		RUN(cudaMalloc((void **)&d_A, N*N*sizeof(float)));
		RUN(cudaMalloc((void **)&d_B, N*N*sizeof(float)));
		RUN(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
		if(N <= max_x)
		{
			dim3 grid(1,1,1);
			dim3 block(N, N, 1);
	      	printf("\nLaunching kernel ");
	      	RUN(cudaEventRecord(startEvent,0));
		  	conv_2D<<<grid,block>>>(d_A, d_B, N);
		  	RUN(cudaEventRecord(stopEvent,0));
		  	RUN(cudaEventSynchronize(stopEvent));
		  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		}
		else
		{
			if(N%max_x == 0)
			{
				dim3 grid(N/max_x, N/max_x, 1);
				dim3 block(max_x,max_x,1);
        		printf("\nLaunching kernel ");
	    		RUN(cudaEventRecord(startEvent,0));
			  	conv_2D<<<grid,block>>>(d_A, d_B, N);
			  	RUN(cudaEventRecord(stopEvent,0));
			  	RUN(cudaEventSynchronize(stopEvent));
			  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
			}
			else
			{
				dim3 grid(N/max_x + 1, N/max_x +1, 1);
				dim3 block(max_x,max_x,1);
        		printf("\nLaunching kernel ");
		    	RUN(cudaEventRecord(startEvent,0));
			  	conv_2D<<<grid,block>>>(d_A, d_B, N);
			  	RUN(cudaEventRecord(stopEvent,0));
			  	RUN(cudaEventSynchronize(stopEvent));
			  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
			}
		}
		
		RUN(cudaGetLastError());
		RUN(cudaMemcpy(h_B, d_B, N*N*sizeof(float), cudaMemcpyDeviceToHost));
		printf("\n Kernel launch complete \n time taken: %.6f ms\n", ms);
		printf("\nPrinting Output:\n");
		for(i=0; i<N; i++)
		{
			for(j=0; j<N; j++)
			{
				printf("%.2f ", h_B[i*N + j]);
			}
			printf("\n");
		}
		printf("\n End of test case: %d\n", k);
		free(h_A);
		free(h_B);
		cudaFree(d_A);
		cudaFree(d_B);
		test_case = test_case -1;
		k = k+1;
		RUN(cudaEventDestroy(startEvent));
		RUN(cudaEventDestroy(stopEvent));
	}
	printf("\n All test cases complete\n");
	return 0;
}