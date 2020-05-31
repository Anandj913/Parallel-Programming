/* -----------------------------------------------------------------------------------------------
Name: Anand Jhunjhunwala
Roll No: 17EC30041
CUDA
Assignment 4: Parallel dotproduct implementation. 
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define thread_per_block 1024

__host__ void RUN(cudaError_t call)
{
	cudaError_t err = call;
	if(err != cudaSuccess)
	{
		fprintf(stderr, " Failed with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__device__ void wrapReduce(float *sdata, int tid, int blockSize)
{
	 if(blockSize >= 64)
	 	{
	 		sdata[tid] = sdata[tid] +  sdata[tid + 32];
	 		__syncthreads();
	 	}
	 if(blockSize >=32)
	 	{
	 		sdata[tid] += sdata[tid + 16];
	 		__syncthreads();
	 	}
	 if(blockSize >=16)
	 	{
	 		sdata[tid] += sdata[tid + 8];
	 		__syncthreads();
	 	}
	 if(blockSize >=8)
	 	{
	 		sdata[tid] += sdata[tid + 4];
	 		__syncthreads();
	 	}
	 if(blockSize >=4)
	 	{
	 		sdata[tid] += sdata[tid + 2];
	 		__syncthreads();
	 	}
	 if(blockSize >=2)
	 		sdata[tid] += sdata[tid + 1];
	 	
}

__global__ void dotproduct(float *gin, float *gout, int N, float *d_A, float *d_B, int flag, int blockSize)
{
	__shared__ float sdata[thread_per_block];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	if(flag == 1)
	{
		if(i<N && (i + blockDim.x) < N)
		{
			sdata[tid] = d_A[i]*d_B[i] + d_A[i + blockDim.x]*d_B[i + blockDim.x];
		}
		else if(i<N)
		{
			sdata[tid] = d_A[i]*d_B[i];
		}
		else
		{
			sdata[tid] = 0;
		}

	}
	else
	{
		if(i<N && (i + blockDim.x) < N)
		{
			sdata[tid] = gin[i] + gin[i + blockDim.x];
		}
		else if(i<N)
		{
			sdata[tid] = gin[i];
		}
		else
		{
			sdata[tid] = 0;
		}
	}
	__syncthreads();
	if(blockSize >= 1024){
		if(tid < 512)
			sdata[tid] = sdata[tid] +  sdata[tid + 512];
		__syncthreads();
	}
	if(blockSize >= 512){
		if(tid < 256)
			sdata[tid] = sdata[tid] +  sdata[tid + 256];
		__syncthreads();
	}
	if(blockSize >= 256){
		if(tid < 128)
			sdata[tid] = sdata[tid] +  sdata[tid + 128];
		__syncthreads();
	}
	if(blockSize >= 128){
		if(tid < 64)
			sdata[tid] = sdata[tid] +  sdata[tid + 64];
		__syncthreads();
	}
	if(tid < 32)
		wrapReduce(sdata, tid, blockSize);
	__syncthreads();
	// writing in global mem
	if(tid == 0)
		gout[blockIdx.x] = sdata[0];
	
}
int main()
{ 
	int test_case, k=1, current_block, call=1;
	long int i, N; 
	float *d_A, *h_A, *d_B, *h_B, *gin, *gout, ms, temp;
	double result=0;
	printf("\n Enter the number of test cases:");
	scanf("%d", &test_case);
	printf(" %d\n", test_case);
	cudaEvent_t startEvent, stopEvent;
	RUN(cudaSetDevice(0));

	while(test_case)
	{
		RUN(cudaEventCreate(&startEvent));
		RUN(cudaEventCreate(&stopEvent));
		printf("\nRunning test case: %d",k);
		printf("\n Enter dimention of vectors:");
		scanf("%ld", &N);
		printf(" %ld\n", N);
		h_A = (float *)malloc(N*sizeof(float));
    	h_B = (float *)malloc(N*sizeof(float));
		printf("\n Enter entries of 1st vector A:\n");
		for(i=0; i<N; i++)
		{
			scanf("%f", &h_A[i]);
		}
		printf("\n Enter entries of 2st vector B:\n");
		for(i=0; i<N; i++)
		{
			scanf("%f", &h_B[i]);
		}
		
		RUN(cudaMalloc((void **)&d_A, N*sizeof(float)));
		RUN(cudaMalloc((void **)&d_B, N*sizeof(float)));
		RUN(cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice));
		RUN(cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice));
		if(N >= 1024)
		{
			current_block = N/(2*thread_per_block);
	    	call = 1;
			while(current_block > 1024)
			{
				current_block = current_block/(2*thread_per_block);
				call = call +1;
			}
			current_block = N;
	    	ms = 0;
			for(i=1; i<=call; i++)
			{
	      		//printf("\n call : %d\n", call);
	      		if(current_block%(2*thread_per_block) == 0)
	      		{
				current_block = current_block/(2*thread_per_block);
				}
				else
				{
					current_block = current_block/(2*thread_per_block);
					current_block++;
				}
	      		//printf("\n current block : %d\n", current_block);
				RUN(cudaMalloc((void **)&gout, current_block*sizeof(float)));
				dim3 grid(current_block, 1,1);
				dim3 block(thread_per_block, 1,1);
		      	RUN(cudaEventRecord(startEvent,0));
				dotproduct<<<grid, block>>>(gin, gout, N, d_A, d_B, i, thread_per_block);
				RUN(cudaEventRecord(stopEvent,0));
			  	RUN(cudaEventSynchronize(stopEvent));
			  	RUN(cudaEventElapsedTime(&temp, startEvent, stopEvent));
			  	ms = ms + temp;
				if(i!=1)
				{
					cudaFree(gin);
				}
				RUN(cudaMalloc((void **)&gin, current_block*sizeof(float)));
	     		RUN(cudaMemcpy(gin, gout, current_block*sizeof(float), cudaMemcpyDeviceToDevice));
				cudaFree(gout);
			}
			RUN(cudaGetLastError());
			//host code to calculate last partial sum 
			free(h_A);
	    	h_A = (float *)malloc(current_block*sizeof(float));
			RUN(cudaMemcpy(h_A, gin, current_block*sizeof(float), cudaMemcpyDeviceToHost)); //tread_per_block == 1024
			cudaFree(gin);
			for(i=0; i<current_block; i++)
			{
				result = result + h_A[i];
			}
			printf("\n Kernel launch complete \n time taken: %.6f ms\n", ms);
			cudaFree(d_A);
			cudaFree(d_B);
			RUN(cudaEventDestroy(startEvent));
			RUN(cudaEventDestroy(stopEvent));
		}
		else
		{
			for(i=0; i<N; i++)
			{
				result = result + h_A[i]*h_B[i];
			}
		}
		printf("\nDot Product of given vectors: %.2f\n", result);
		printf("\n End of test case: %d\n", k);
		free(h_A);
		free(h_B);
		result = 0;
		test_case = test_case -1;
		k = k+1;
	}
	printf("\n All test cases complete\n");
	return 0;
}