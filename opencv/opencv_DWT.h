#pragma once
#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
//系统包
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"
#include "opencv_DFT.h"
using namespace std;
using namespace cv;
void base_code(char *path, float* h, int len);

