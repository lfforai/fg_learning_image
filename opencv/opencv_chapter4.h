#pragma once
#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
#include  "image_rotate.cuh"
//系统包
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"
#include<time.h>

void chapter4();