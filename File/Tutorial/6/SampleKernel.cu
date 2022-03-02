#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#define SIZE 3
//__device__ int manage;
__device__ __managed__ int manage;

__global__ void mykernel(int *test_d)
{
	int i;

	printf("In kernel! test_d is ");
	for (i = 0; i < SIZE; i++)
	{
		printf("%d", test_d[i]);
	}
	printf("\n");

	for (i = 0; i < SIZE; i++)
	{
		test_d[i]=9;
	}

	printf("In kernel! test_d is updated as ");
	for (i = 0; i < SIZE; i++)
	{
		printf("%d", test_d[i]);
	}
	printf("\n");

	printf("In kernel: manage is %d\n", manage);
	manage = 2;
}


int main()
{
	cudaError_t cudaStatus; 
	int * test_h;
	int * test_d;
	int i; 
	manage = 1;

	cudaSetDevice(0);
	cudaMalloc(&test_d, sizeof(int)*SIZE);
	test_h = (int *)malloc(sizeof(int)*SIZE);

	for (i = 0; i < SIZE; i++)
	{
		test_h[i] = 0;
	}
	printf("\n");

	cudaMemcpy(test_d, test_h, sizeof(int)*SIZE, cudaMemcpyHostToDevice);

	mykernel << <1, 1 >> > (test_d);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "my kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaMemcpy(test_h, test_d, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);

	printf("After kernel, test_h is ");
	for (i = 0; i < SIZE; i++)
	{
		printf("%d", test_h[i]);
	}
	printf("\n");

	cudaFree(test_d);
	free(test_h);

	cudaDeviceSynchronize();
	cudaDeviceReset();


	printf("In host: manage is %d\n", manage);

	return 0;

}
