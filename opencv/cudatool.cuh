#pragma once
struct grid_block_size {
	int blockSize;
	int minGridSize;
};

//grid_block_size* bestBlockSize_image(void (*maykernel)( ),int blockSizeLimit,int u_blocksize,int u_gridsize)