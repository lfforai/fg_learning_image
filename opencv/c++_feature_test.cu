#include "c++_feature_test.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <bitset> 

#include <stdio.h>

#define THREADS_PER_BLOCK          256
#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
#define MY_KERNEL_MIN_BLOCKS   3
#else
#define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS   2
#endif

void BinaryBitset(int n)
{
	cout<<bitset<sizeof(int) * 8>(n)<<endl;
}

__global__ void vote_all(int *a, int *b, int n)
{
	int tid = threadIdx.x;
	if (tid > n)
	{
		return;
	}
	int temp = a[tid];
	b[tid] = __all_sync(0xffffffff, temp > 100);
}

__global__ void vote_any(int *a, int *b, int n)
{
	int tid = threadIdx.x;
	if (tid > n)
	{
		return;
	}
	int temp = a[tid];
	b[tid] = __any_sync(0xffffffff, temp > 100);
}

__global__ void vote_ballot(int *a, int *b, int n)
{
	int tid = threadIdx.x;
	if (tid > n)
	{
		return;
	}
	int temp = a[tid];
	
	b[tid] = __ballot_sync(0xffffffff, temp > 100);
}

__global__ void activemask(int *a, int *b, int n)
{
	int tid = threadIdx.x;
	if (tid > n)
	{
		return;
	}

	b[tid] = __activemask();
}

int test_feature()
{
	int *h_a, *h_b, *d_a, *d_b;
	int n = 256, m = 10;
	int nsize = n * sizeof(int);
	h_a = (int *)malloc(nsize);
	h_b = (int *)malloc(nsize);
	int vote = 0;
	for (int i = 0; i < n; ++i)
	{
		h_a[i] = i;
		//cout << h_a[i] << endl;
	}
	
	memset(h_b, 0, nsize);
	cudaMalloc(&d_a, nsize);
	cudaMalloc(&d_b, nsize);
	cudaMemcpy(d_a, h_a, nsize, cudaMemcpyHostToDevice);
	cudaMemset(d_b, 0, nsize);
	vote_all << <1, 256 >> > (d_a, d_b, n);
	cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
	printf("vote_all():");
	for (int i = 0; i < n; ++i)
	{
		if (!(i % m))
		{
			printf("\n");
		}
		if (h_b[i] == 0)
			vote += 1;
		printf("%d", h_b[i]);
	}
	printf("\n");
	cout << "vote-----" << vote << endl;
	vote = 0;
	vote_any << <1, 256 >> > (d_a, d_b, n);
	cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
	printf("vote_any():");
	for (int i = 0; i < n; ++i)
	{
		if (!(i % m))
		{
			printf("\n");
		}
		printf("%d", h_b[i]);
		if (h_b[i] == 0)
			vote += 1;
	}
	printf("\n");
	cout << "vote-----" << vote << endl;
	vote_ballot << <1, 256 >> > (d_a, d_b, n);
	cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
	vote = 0;
	printf("vote_ballot():");
	for (int i = 0; i < n; ++i)
	{
		if (!(i % m))
		{
			printf("\n");
		}
		if (h_b[i] == 0)
			vote += 1;
		printf(",%d", (uint)h_b[i]);
	}
	printf("\n");
	cout << "vote-----" << vote << endl;

	vote = 0;
	activemask << <1, 256 >> > (d_a, d_b, n);
	cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
	vote = 0;
	printf("activemask():");
	for (int i = 0; i < n; ++i)
	{
		if (!(i % m))
		{
			printf("\n");
		}
		if (h_b[i] == 0)
			vote += 1;
		printf(",%d", (uint)h_b[i]);
	}
	printf("\n");
	cout << "vote-----" << vote << endl;

	BinaryBitset(-32);
	BinaryBitset(-1);
	return 0;
}
