/* -----------------------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll No: 17EC30041
CUDA
Assignment 3: Matrix transpose using dynamic shared memory
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define cal_per_thread 4 //define calculation performend by each thred,
					     //keep it in power of 2

__host__ void RUN(cudaError_t call)
{
	cudaError_t err = call;
	if(err != cudaSuccess)
	{
		fprintf(stderr, " Failed with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
__global__ void mat_transpose(float *d_A, float *d_B, int N, int tile_dim, int block_row)
{
	extern __shared__ float tile[];

	unsigned int x = blockIdx.x*tile_dim + threadIdx.x;
	unsigned int y = blockIdx.y*tile_dim + threadIdx.y;

	for(int j=0; j<tile_dim; j+= block_row)
	{
		if(y+j < N && x < N)
		tile[(threadIdx.y + j)*(tile_dim + 1) + threadIdx.x] = d_A[(y+j)*N + x];
	}

	__syncthreads();
 

	x = blockIdx.y*tile_dim + threadIdx.x;
	y = blockIdx.x*tile_dim + threadIdx.y;

	for(int j=0; j<tile_dim; j+= block_row)
	{
		if(y+j < N && x<N)
		d_B[(y+j)*N + x]= tile[threadIdx.x*(tile_dim + 1) + threadIdx.y + j];
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

	cudaEvent_t startEvent, stopEvent;
	cudaDeviceProp devp;
	RUN(cudaGetDeviceProperties(&devp, 0));
	int shared_mem_size = devp.sharedMemPerBlock;
	RUN(cudaSetDevice(0));

	shared_mem_size = shared_mem_size/(sizeof(float));
	shared_mem_size = sqrt(shared_mem_size);
	if(shared_mem_size < max_x)
	{	
		printf("\n Not enough shared memory space available \n");
		printf("Please reduce max_x and try again\n");
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
		  	mat_transpose<<<grid,block, N*(N+1)*sizeof(float)>>>(d_A, d_B, N, N, N);
		  	RUN(cudaEventRecord(stopEvent,0));
		  	RUN(cudaEventSynchronize(stopEvent));
		  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		}
		else
		{
			if(N%(max_x) == 0)
			{
				dim3 grid(N/(max_x), N/(max_x), 1);
				dim3 block(max_x,max_x/cal_per_thread,1);
        		printf("\nLaunching kernel ");
	      		RUN(cudaEventRecord(startEvent,0));
		    	mat_transpose<<<grid,block, max_x*(max_x+1)*sizeof(float)>>>(d_A, d_B, N, max_x, max_x/cal_per_thread);
		    	RUN(cudaEventRecord(stopEvent,0));
			  	RUN(cudaEventSynchronize(stopEvent));
			  	RUN(cudaEventElapsedTime(&ms, startEvent, stopEvent));
			}
			else
			{
				dim3 grid(N/max_x +1, N/max_x + 1, 1);
				dim3 block(max_x,max_x/cal_per_thread,1);
        		printf("\nLaunching kernel ");
        		RUN(cudaEventRecord(startEvent,0));
		    	mat_transpose<<<grid,block, max_x*(max_x+1)*sizeof(float)>>>(d_A, d_B, N, max_x, max_x/cal_per_thread);
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