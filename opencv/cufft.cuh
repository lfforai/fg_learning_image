#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "math.h"
#include <device_launch_parameters.h>
#include <sstream>
#include <curand.h>
#include "cufft.h"
#include "math.h"
using namespace std;
using namespace cv;

extern "C" void cufft(char* path , int mode);
void cuffttest(char *path);

//Õ®”√À„◊”
extern "C" Mat fre_spectrum(cufftComplex* data, int NX, int NY, int mode);
extern "C" cufftComplex* cufft_fun(const char* path, Mat Lena_o, int mode = 0, int m_mode = 0, int MN = 0,int ifgray=0);
extern "C" Mat angle_spectrum(cufftComplex* data, int NX, int NY);