#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include "math.h"
#include <device_launch_parameters.h>
#include <sstream>
#include <curand.h>
#include "cufft.h"
#include "opencv_DFT.h"

//// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//
//// CUDA helper functions
//#include <helper_cuda.h>         // helper functions for CUDA error check
using namespace std;
using namespace cv;


__device__ __host__  struct Point_gpu {
	float x;
	float y;
};

//二值化版
struct se_tpye {
	uchar* data;//全部为1的向量,目前没有用到为灰度形态预留
	Point_gpu* point_offset;//相对于中心的位置的偏移量
	Point_gpu center;//目前没用
	int length;
	void init(int length_N, Point_gpu*  point_offset_N,uchar* data_N)
	{
		this->length = length_N;
		//检测cuda设备
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount > 0)
		{
			cudaMallocManaged((void**)&point_offset, sizeof(Point_gpu)*length);
			cudaMallocManaged((void**)&data, sizeof(uchar)*length);

			this->center.x = 0.0;//cols
			this->center.y = 0.0;//rows
			cudaMemcpy(point_offset, point_offset_N, sizeof(Point_gpu)*length,cudaMemcpyDefault);
			cudaMemcpy(data, data_N,  sizeof(uchar)*length, cudaMemcpyDefault);
		/*	for (size_t i = 0; i < length; i++)
			{
				cout<<point_offset[i].y<<endl;
				cout<<point_offset[i].x<< endl;
				printf("value:%u \n",data[i]);
				cout<<"---------------"<<endl;
			}*/
		}
		else {
			//目前没有cpu版
			cout<<"gpu no use!"<<endl;
		}
	}
};

//灰度版
struct se_tpye_gray {
	int* data;//全部为1的向量,目前没有用到为灰度形态预留
	Point_gpu* point_offset;//相对于中心的位置的偏移量
	Point_gpu center;//目前没用
	int length;
	void init(int length_N, Point_gpu*  point_offset_N, int* data_N)
	{
		this->length = length_N;
		//检测cuda设备
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount > 0)
		{
			cudaMallocManaged((void**)&point_offset, sizeof(Point_gpu)*length);
			cudaMallocManaged((void**)&data, sizeof(int)*length);

			this->center.x = 0.0;//cols
			this->center.y = 0.0;//rows
			cudaMemcpy(point_offset, point_offset_N, sizeof(Point_gpu)*length, cudaMemcpyDefault);
			cudaMemcpy(data, data_N, sizeof(int)*length, cudaMemcpyDefault);
		}
		else {
			//目前没有cpu版
			cout << "gpu no use!" << endl;
		}
	}
};

Mat AND_two(const Mat& A, const Mat& B, uchar min = 0, uchar max = 255);
Mat OR_two(const Mat& A, const Mat& B, uchar min = 0, uchar max = 255);
Mat NOT_two(const Mat& A, uchar min = 0, uchar max = 255);
Mat AND_NOT_two(const Mat& A, const Mat& B, uchar min = 0, uchar max = 255);
Mat XOR_two(const Mat& A, const Mat& B, uchar min = 0, uchar max = 255);
Point_gpu* set_Point_gpu(int M, int N);
void morphology_test(int M, int N,int mode);
void chapter9();

template<class T>
struct filter_screem {
	T *data;
	Point_gpu* postion;
	int len;
	void init(int M, int N)
	{
		len = N * M;
		cudaMallocManaged((void **)&data, sizeof(T)*N*M);
		cudaMallocManaged((void **)&postion, sizeof(Point_gpu)*N*M);
		int M_center = (int)M / 2;//y
		int N_center = (int)N / 2;//x
		for (size_t i = 0; i < M; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				data[i*N + j] = 1;
				postion[i*N + j].x = (int)j - N_center;
				postion[i*N + j].y = (int)i - M_center;
			}
		}
	}
};


void chapter10_test();