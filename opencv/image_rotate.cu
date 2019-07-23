#pragma once
#include "image_rotate.cuh"

//一、旋转变换
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
	//imshow("原图：", Lena);
	//imshow("旋转以后的图：", dstImg1);
	return dstImg1.clone();
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	/*waitKey(0);*/
}

//二、坐标点移动
// Texture reference for 2D float texture
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_move;

//声明CUDA数组
cudaArray* cuArray_move;//用于计算最近point插值

__global__ void transformKernel_move(uchar *outputData,
	int height,
	int width,
	int x_move,
	int y_move)
{
	// calculate normalized texture coordinates
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int tu = x - x_move;
	int tv = y - y_move;

	// read from texture and write to global memory
	outputData[y*width + x] = tex2D(refTex_move,tu,tv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
Mat image_move_point(char* path, Mat lena_o, int ifhd, int x_move,int y_move)
{     
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
	refTex.normalized = false;
	refTex.filterMode = cudaFilterModePoint;
	//绑定cuArray到纹理
	cudaMallocArray(&cuArray_move, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_move, cuArray_move);
	//拷贝数据到cudaArray
	t = cudaMemcpyToArray(cuArray_move, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//输出放缩以后在cpu上图像
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

	//输出放缩以后在cuda上的图像
	uchar* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

	dim3 block(8, 8);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	transformKernel_move << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less,x_move,y_move);
	cudaDeviceSynchronize();

	//从GPU拷贝输出数据到CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_move);
	cudaFree(pDstImgData1);
	//imshow("原图：", Lena);
	//imshow("移动以后的图：", dstImg1);
	return dstImg1.clone();
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	/*waitKey(0);*/
}

Mat image_rotate_point_GPU(char* path, Mat lena_o, int ifhd) {
	return image_rotate_point(path, lena_o, ifhd);
}

Mat image_move_point_GPU(char* path, Mat lena_o, int ifhd, int x_move,int y_move) {
	return image_move_point(path, lena_o,ifhd,x_move,y_move);
}

texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_Laplace;

cudaArray* cuArray_Laplace;

cudaChannelFormatDesc cuDesc_Laplace = cudaCreateChannelDesc<uchar>();

//拉普拉斯算子
	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
__global__ void Kerkel_Laplace(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, int mode, int c)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		if (mode == 0)
		{
			pDstImgData[idx] =(int)((tex2D(refTex_Laplace, tidx + 1, tidy) + tex2D(refTex_Laplace, tidx - 1, tidy) + tex2D(refTex_Laplace, tidx, tidy + 1) + tex2D(refTex_Laplace, tidx, tidy - 1)) - (int)4 * tex2D(refTex_Laplace, tidx, tidy))*c;
		}

		if (mode == 1)
		{
			pDstImgData[idx]=(int)((tex2D(refTex_Laplace, tidx + 1, tidy) + tex2D(refTex_Laplace, tidx - 1, tidy)
				+ tex2D(refTex_Laplace, tidx, tidy + 1) + tex2D(refTex_Laplace, tidx, tidy - 1) +
				tex2D(refTex_Laplace, tidx - 1, tidy - 1) + tex2D(refTex_Laplace, tidx + 1, tidy + 1) +
				tex2D(refTex_Laplace, tidx + 1, tidy - 1) + tex2D(refTex_Laplace, tidx - 1, tidy + 1)) - (int)8 * tex2D(refTex_Laplace, tidx, tidy))*c;
		}//printf("value=%u,%d,%d,%f,%f \n", pdstimgdata[idx], x1, y2, x_des, y_des);

		if (mode == 3)
		{
			pDstImgData[idx] = (int)(((0)*(int)tex2D(refTex_Laplace, tidx + 1, tidy) + (0)*(int)tex2D(refTex_Laplace, tidx - 1, tidy)
				+2* (int)tex2D(refTex_Laplace, tidx, tidy + 1) -2* (int)tex2D(refTex_Laplace, tidx, tidy - 1) 
				-1*(int)tex2D(refTex_Laplace, tidx - 1, tidy - 1) + 1* (int)tex2D(refTex_Laplace, tidx + 1, tidy + 1) 
				-1* (int)tex2D(refTex_Laplace, tidx + 1, tidy - 1) + 1*(int)tex2D(refTex_Laplace, tidx - 1, tidy + 1)) - 0 * (int)tex2D(refTex_Laplace, tidx, tidy))*c;
		
	/*		pDstImgData[idx] = (int)(((0)*(int)tex2D(refTex_Laplace, tidx + 1, tidy) + (0)*(int)tex2D(refTex_Laplace, tidx - 1, tidy)
				 -2 * (int)tex2D(refTex_Laplace, tidx, tidy + 1) + 2 * (int)tex2D(refTex_Laplace, tidx, tidy - 1) +
				1*(int)tex2D(refTex_Laplace, tidx - 1, tidy - 1) -1 * (int)tex2D(refTex_Laplace, tidx + 1, tidy + 1) +
				1 * (int)tex2D(refTex_Laplace, tidx + 1, tidy - 1) -(1)*(int)tex2D(refTex_Laplace, tidx - 1, tidy + 1)) - 0 * (int)tex2D(refTex_Laplace, tidx, tidy))*c;*/
		}//printf("value=%u,%d,%d,%f,%f \n", pdstimgdata[idx], x1, y2, x_des, y_des);
	}
}

//图像进行标定
void demarcate(Mat& image_src)
{
	Mat image = image_src.clone();
	image.convertTo(image, CV_32FC1);
	double minVal, maxVal;
	int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
	minMaxIdx(image, &minVal, &maxVal, minIdx, maxIdx);
	cout << minVal << endl;
	cout << maxVal << endl;
	image=image - minVal;
	image = image / maxVal;
	image = image * 255;
	image.convertTo(image, CV_8U);
	image_src=image.clone();
	//cout<<image_src<<endl;
}

//mode=0,1 c=1,-1
void Laplace_cuda(Mat &image, int mode, int c) {

	////只处理灰度值
	Mat Lena = image.clone();
	int imgWidth_src = Lena.cols;//原图像宽
	int imgHeight_src = Lena.rows;//原图像高
	int channels = Lena.channels();

	int x_rato_less = 1;
	int y_rato_less = 1;

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

	//设置1纹理属性
	cudaError_t t;
	refTex_Laplace.addressMode[0] = cudaAddressModeBorder;
	refTex_Laplace.addressMode[1] = cudaAddressModeBorder;
	refTex_Laplace.normalized = false;
	refTex_Laplace.filterMode = cudaFilterModePoint;
	//绑定cuArray到纹理
	cudaMallocArray(&cuArray_Laplace, &cuDesc_Laplace, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_Laplace, cuArray_Laplace);
	//拷贝数据到cudaArray
	t = cudaMemcpyToArray(cuArray_Laplace, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//输出放缩以后在cpu上图像
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//缩小

	//输出放缩以后在cuda上的图像
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	Kerkel_Laplace <<<grid, block >>> (pDstImgData1, imgHeight_des_less, imgWidth_des_less ,mode ,c);
	cudaDeviceSynchronize();

	//从GPU拷贝输出数据到CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_Laplace);
	cudaFree(pDstImgData1);
	image=dstImg1.clone();
	//cout << image << endl;
	//cout<<image<<endl;
	//image.convertTo(image,CV_8U);
}