#pragma once
#include "image_rotate.cuh"

// Texture reference for 2D float texture
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex;

//声明CUDA数组
cudaArray* cuArray;//用于计算最近point插值

//通道数
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar>();

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(uchar *outputData,
	int height, 
	int width,
	const float theta)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float u = (float)x - (float)width  / 2.0;
	float v = (float)y - (float)height / 2.0;
	float tu = u * cosf(theta) - v * sinf(theta);
	float tv = v * cosf(theta) + u * sinf(theta);

	tu /= (float)width;
	tv /= (float)height;

	// read from texture and write to global memory
	outputData[y*width + x] = (uchar)(tex2D(refTex, tu + 0.5f, tv + 0.5f));
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
Mat image_rotate_point(char* path,Mat lena_o,int ifhd)
{
	const float angle = -0.5f;        // angle to rotate image by (in radians)
	Mat Lena;
	if (strlen(path) == 0) {
		Lena = lena_o.clone();
	}
	else {
		Lena = imread(path);
	}

	if (ifhd == 0)//不是灰度图要进行转换
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

	float x_rato_less = 1.0;
	float y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//原图像宽
	int imgHeight_src = Lena.rows;//原图像高
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

   //设置1纹理属性
	cudaError_t t;
	refTex.addressMode[0] = cudaAddressModeBorder;
	refTex.addressMode[1] = cudaAddressModeBorder;
	refTex.normalized = true;
	refTex.filterMode = cudaFilterModePoint;
	//绑定cuArray到纹理
	cudaMallocArray(&cuArray, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex, cuArray);
	//拷贝数据到cudaArray
	t = cudaMemcpyToArray(cuArray, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//输出放缩以后在cpu上图像
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

	//输出放缩以后在cuda上的图像
	uchar* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

	dim3 block(8, 8);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	transformKernel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, angle);
	cudaDeviceSynchronize();

	//从GPU拷贝输出数据到CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray);
	cudaFree(pDstImgData1);
	imshow("原图：", Lena);
	imshow("旋转以后的图：", dstImg1);
	return dstImg1.clone();
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	/*waitKey(0);*/
}

Mat image_rotate_point_GPU(char* path, Mat lena_o, int ifhd) {
	return image_rotate_point(path, lena_o, ifhd);
}