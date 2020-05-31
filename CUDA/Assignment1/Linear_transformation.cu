/**
Name: Anand Jhunjhunwala
Roll Number: 17EC30041
Assignment 1: Linear Transformation 
**/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining kernels as specified from assignment 
__global__ void process_kernel1(float *input1, float *input2, float *output_k1, int datasize)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*(blockDim.x) + threadIdx.x;
	int i = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;
	if(i<datasize)
	{
		output_k1[i] = sin(input1[i]) + cos(input2[i]);
	}

}

__global__ void process_kernel2(float *input, float *output_k2, int datasize)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*(blockDim.x) + threadIdx.x;
	int i = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;
	if(i<datasize)
	{
		output_k2[i] = log(input[i]);
	}

}

__global__ void process_kernel3(float *input, float *output_k3, int datasize)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*(blockDim.x) + threadIdx.x;
	int i = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;
	if(i<datasize)
	{
		output_k3[i] = sqrt(input[i]);
	}

}

int main(void)
{
	cudaError_t err = cudaSuccess;

	int size = pow(2,14);
	int total,i; 

    // host memory allocation 
	float *h_input1 = (float *)malloc(size*sizeof(float));
	float *h_input2 = (float *)malloc(size*sizeof(float));
	float *h_output1 = (float *)malloc(size*sizeof(float));
	float *h_output2 = (float *)malloc(size*sizeof(float));
	float *h_output3 = (float *)malloc(size*sizeof(float));
	if (h_input1 == NULL || h_input2 == NULL || h_output3 == NULL || h_output2 == NULL || h_output1 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    else
    {
    	printf("Host memory allocation complete. \n");
    }

    // Data input from the user
    printf("Enter first vector of size 2^14\n");
	for(i=0;i<size;i++)
	{
		scanf("%f",&h_input1[i]);
	}
	printf("Enter second vector of size 2^14\n");
	for(i=0;i<size;i++)
	{
		scanf("%f",&h_input2[i]);
	}
	// printf("Enter total number of elements in a vector\n");
	// scanf("%d",&total);

 //    // If total number of data is not 2^14 then exit the code
	// if(size != total)
	// {
	// 	printf("Input size is unequal to specified value\n");
	// 	exit(0);
	// }

    //device memory allocation and data copy for kernel 1
	float *d_input1 = NULL; 
	err = cudaMalloc((void **)&d_input1, size*sizeof(float));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device input 1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	float *d_input2 = NULL;
	err = cudaMalloc((void **)&d_input2, size*sizeof(float));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device input 2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	float *d_output1 = NULL;
	err = cudaMalloc((void **)&d_output1, size*sizeof(float));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device output 1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_input1, h_input1, size*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, h_input2, size*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //kernel 1 call 
    dim3 grid1(4,2,2);
    dim3 block1(32,32,1);
    printf("Launching kernel 1\n");
    process_kernel1<<<grid1, block1>>>(d_input1, d_input2, d_output1, size);
    err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else
    {
    	printf("process_kernel1 launched successfully\n");
    }

    //copying processed data of kernel 1 from device to host 
    err = cudaMemcpy(h_output1, d_output1, size*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //device memory allocation and data copy for kernel 2
    float *d_output2 = NULL;
	err = cudaMalloc((void **)&d_output2, size*sizeof(float));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device output 2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    //kernel 2 call
	dim3 grid2(2,16,1);
    dim3 block2(8,8,8);
    printf("Launching kernel 2\n");
    process_kernel2<<<grid2, block2>>>(d_output1, d_output2, size);
    err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else
    {
    	printf("process_kernel2 launched successfully\n");
    }

    //copying processed data of kernel 2 from device to host 
    err = cudaMemcpy(h_output2, d_output2, size*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //device memory allocation and data copy for kernel 3
    float *d_output3 = NULL;
	err = cudaMalloc((void **)&d_output3, size*sizeof(float));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device output 3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    //kernel 3 call
	dim3 grid3(16,1,1);
    dim3 block3(128,8,1);
    printf("Launching kernel 3\n");
    process_kernel3<<<grid3, block3>>>(d_output2, d_output3, size);
    err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else
    {
    	printf("process_kernel3 launched successfully\n");
    }

    //copying processed data of kernel 3 from device to host 
    err = cudaMemcpy(h_output3, d_output3, size*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output3 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Printing the processed result 
    printf("Printing the result of kernels\n");
    // printf("Result of kernel 1\n");
    // for(i=0;i<size;i++)
    // {
    // 	printf("%.2f ", h_output1[i]);
    // }
    // printf("\nResult of kernel 2\n");
    // for(i=0;i<size;i++)
    // {
    // 	printf("%.2f ", h_output2[i]);
    // }
    printf("\nResult of kernel 3\n");
    for(i=0;i<size;i++)
    {
    	printf("%.2f ", h_output3[i]);
    }

    // Free device memory
    err = cudaFree(d_input1);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_input2);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_output1);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_output2);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_output3);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("\nDevice memory successfully freed\n");

    //Free Host Memory
    free(h_input1);
    free(h_input2);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    printf("Host memory successfully freed\n");

    //Reset Device
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}