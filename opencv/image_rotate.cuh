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
//// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//
//// CUDA helper functions
//#include <helper_cuda.h>         // helper functions for CUDA error check
using namespace std;
using namespace cv;

Mat image_rotate_point_GPU(char* path, Mat lena_o, int ifhd=0);
Mat image_move_point_GPU(char* path, Mat lena_o, int ifhd, int x_move, int y_move);
void Laplace_cuda(Mat &image, int mode, int c);
void demarcate(Mat& image_src);