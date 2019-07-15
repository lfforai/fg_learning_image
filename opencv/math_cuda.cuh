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

void atan_cpu_test();
void cufft_math_test(char* path, int mode = 0);