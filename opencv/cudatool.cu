#include "cudatool.cuh"
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
#include "cuda_runtime.h"

//// Host code arrayCount=blockSizeLimit 
//grid_block_size* bestBlockSize_image(void (*maykernel)( ),int blockSizeLimit,int u_blocksize,int u_gridsize)
//{
//	int blockSize;      // The launch configurator returned block size
//	int minGridSize;    // The minimum grid size needed to achieve the
//						// maximum occupancy for a full device
//						// launch
//	grid_block_size* result = (grid_block_size *)malloc(sizeof(grid_block_size));
//	cudaOccupancyMaxPotentialBlockSize(
//		&minGridSize,
//		&blockSize,
//		(void*)maykernel,
//		0,
//		blockSizeLimit);
//	result->minGridSize = minGridSize;
//	result->blockSize = blockSize;
//	return result;
//}

int numBlocks;        // Occupancy in terms of active blocks
int blockSize = 32;

// These variables are used to convert occupancy to warps
//int device;
//cudaDeviceProp prop;
//int activeWarps;
//int maxWarps;
//
//cudaGetDevice(&device);
//cudaGetDeviceProperties(&prop, device);
//
//cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//	&numBlocks,
//	MyKernel,
//	blockSize,
//	0);
//
//activeWarps = numBlocks * blockSize / prop.warpSize;
//maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
//
//std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
//
//return 0;