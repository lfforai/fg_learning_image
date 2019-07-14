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
#include "cudatool.cuh"

namespace image_scale0 {
	extern "C" int image_scale0();
}
namespace image_scale1 {
	extern "C" int image_scale1();
}

namespace image_scale2 {
	extern "C" int image_scale2();
}