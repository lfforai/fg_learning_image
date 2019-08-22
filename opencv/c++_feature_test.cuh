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
int test_feature();