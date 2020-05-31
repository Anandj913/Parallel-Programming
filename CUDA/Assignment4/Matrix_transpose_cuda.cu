/* -----------------------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll No: 17EC30041
CUDA
Assignment 2: Matrix transpose using rectangular tile
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define W 32 //tile dimention W*2W please keep it in multiple of 32

__host__ void RUN(cudaError_t call)
{
	cudaError_t err = call;
	if(err != cudaSuccess)
	{
		fprintf(stderr, " Failed with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
__global__ void mat_transpose(float *d_A, float *d_B, int height, int width)
{
	unsigned int x = blockIdx.x*2*W + threadIdx.x;
	unsigned int y = blockIdx.y*W + threadIdx.y;

	__shared__ float tile[W][2*W + 1];



	for(int j=0; j<2*blockDim.x*(W/blockDim.x); j+= blockDim.x)
	{
		for(int i=0; i<blockDim.y*(W/blockDim.y); i+= blockDim.y)
		{
			if(x+j < width && y+i < height)
			tile[threadIdx.y + i][threadIdx.x + j] = d_A[(y+i)*width + x + j];
		}
	}

	__syncthreads();


	x = blockIdx.y*W + threadIdx.x;
	y = blockIdx.x*2*W + threadIdx.y;

	for(int j=0; j<2*blockDim.x*(W/blockDim.x); j+= blockDim.x)
	{
		for(int i=0; i<blockDim.y*(W/blockDim.y); i+= blockDim.y)
		{
			if(y+j < width && x+i<height)
			d_B[(y+j)*height + x + i]= tile[threadIdx.x+i][threadIdx.y + j];
		}
	}
}
int main()
{
	int max_x = 32; //maximum_thread_per_block 1024
	int test_case, N, k=1;
	long int i,j; 
	float *d_A, *h_A, *d_B, *h_B, ms;
	printf("\n Enter the number of test cases:");
	scanf("%d", &test_case);
	printf(" %d\n", test_case);

	cudaDeviceProp devp;
	cudaEvent_t startEvent, stopEvent;
	RUN(cudaGetDeviceProperties(&devp, 0));
	int shared_mem_size = devp.sharedMemPerBlock;
	RUN(cudaSetDevice(0));

	shared_mem_size = shared_mem_size/(2*sizeof(float));
	shared_mem_size = sqrt(shared_mem_size);
	if(shared_mem_size < W)
	{	
		printf("\n Not enough shared memory space available \n");
		printf("Please reduce W and try again\n");
		exit(EXIT_FAILURE);
	}
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
		  	mat_transpose<<<grid,block>>>(d_A, d_B, N, N);
		  	RUN(cudaEventRecord(stopEvent,0));
		  	RUN(cudaEventSynchronize(stopEvent));
		  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));

		}
		else
		{
			if(N%(2*W) == 0)
			{
				dim3 grid(N/(2*W), N/(W), 1);
				dim3 block(max_x,max_x,1);
        		printf("\nLaunching kernel ");
	    		RUN(cudaEventRecord(startEvent,0));
			  	mat_transpose<<<grid,block>>>(d_A, d_B, N, N);
			  	RUN(cudaEventRecord(stopEvent,0));
			  	RUN(cudaEventSynchronize(stopEvent));
			  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
			}
			else
			{
				dim3 grid(N/(2*W) +1, N/W, 1);
				dim3 block(max_x,max_x,1);
        		printf("\nLaunching kernel ");
		    	RUN(cudaEventRecord(startEvent,0));
			  	mat_transpose<<<grid,block>>>(d_A, d_B, N, N);
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
			for(j=0; j<N; j++)		{
				printf("%.2f ", h_B[i*N + j]);
			}
			printf("\n");
		}
		printf("\n End of test case: %d\n", k);
		ms =0;
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