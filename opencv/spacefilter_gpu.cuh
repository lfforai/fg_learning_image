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
#include <typeinfo>
#include "opencv_chapter9.cuh"
#include <time.h>

using namespace std;
using namespace cv;

__device__ __host__  struct Point_f{
	float x;
	float y;
};

template<class datatype>
struct f_screem {
	datatype *data;
	Point_f* postion;
	int len;
	void init(int M, int N)
	{
		len = N * M;
		cudaMallocManaged((void **)&data, sizeof(datatype)*N*M);
		cudaMallocManaged((void **)&postion, sizeof(Point_f)*N*M);
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

enum sf_mode {
	prewitt_x_N = 0,
	prewitt_y_N = 1,

	sobel_x_N = 2,
	sobel_y_N = 3,

	sobel_45z_N = 4,
	sobel_45f_N = 5,

	Laplace8_N = 6,
	Gauss25_N = 7,
	LoG_N = 8,
	avg_5=9 //5*5¾ùÖµÄ£°å
};


template<class datatype, class arraytype>
Mat space_filter_gpu(char * path, Mat& image, int len, Point_f*  point_offset_N, datatype* data, float size);

template<class datatype>
f_screem<datatype>* set_f(sf_mode mode);

float Max_ofmat(Mat& H);
Mat sobel_grad(Mat& image_N,int mode);

void LoG_test();
void canny_test();
void Partial_treatment();