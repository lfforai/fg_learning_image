#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
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

struct se_tpye {
	uchar* data;//ȫ��Ϊ1������,Ŀǰû���õ�Ϊ�Ҷ���̬Ԥ��
	Point_gpu* point_offset;//��������ĵ�λ�õ�ƫ����
	Point_gpu center;//Ŀǰû��
	int length;
	void init(int length_N, Point_gpu*  point_offset_N,uchar* data_N)
	{
		this->length = length_N;
		//���cuda�豸
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount > 0)
		{
			cout<<"---------------------"<<endl;
			cudaMallocManaged((void**)&point_offset, sizeof(Point_gpu)*length);
			cudaMallocManaged((void**)&data, sizeof(uchar)*length);
	/*		for (size_t i = 0; i < length; i++)
			{
				data[i] = 255;
			}*/
			this->center.x = 0.0;//cols
			this->center.y = 0.0;//rows
			cudaMemcpy(point_offset, point_offset_N, sizeof(Point_gpu)*length,cudaMemcpyDefault);
			cudaMemcpy(data, data_N,  sizeof(uchar)*length, cudaMemcpyDefault);
			for (size_t i = 0; i < length; i++)
			{
				cout<<point_offset[i].y<<endl;
				cout<<point_offset[i].x<< endl;
				printf("%u \n",data[i]);
				cout<<"---------------"<<endl;
			}
		}
		else {
			//Ŀǰû��cpu��
		}
	}
};

Point_gpu* set_Point_gpu(int M, int N);
void morphology_test(int M, int N,int mode);