#pragma once
#include "cufft.cuh"        //����Ҷ�˲�ʽ��
#include "image_scale.cuh"  //ͼ������ʵ��
//ϵͳ��
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"
#include "opencv_DFT.h"
using namespace std;
using namespace cv;
void base_code(char *path, float* h, int len);

