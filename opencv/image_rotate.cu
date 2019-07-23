#pragma once
#include "image_rotate.cuh"

//һ����ת�任
// Texture reference for 2D float texture
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex;

//����CUDA����
cudaArray* cuArray;//���ڼ������point��ֵ

//ͨ����
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

	if (ifhd == 0)//���ǻҶ�ͼҪ����ת��
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	float x_rato_less = 1.0;
	float y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

   //����1��������
	cudaError_t t;
	refTex.addressMode[0] = cudaAddressModeBorder;
	refTex.addressMode[1] = cudaAddressModeBorder;
	refTex.normalized = true;
	refTex.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex, cuArray);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	uchar* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

	dim3 block(8, 8);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	transformKernel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, angle);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray);
	cudaFree(pDstImgData1);
	//imshow("ԭͼ��", Lena);
	//imshow("��ת�Ժ��ͼ��", dstImg1);
	return dstImg1.clone();
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	/*waitKey(0);*/
}

//����������ƶ�
// Texture reference for 2D float texture
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_move;

//����CUDA����
cudaArray* cuArray_move;//���ڼ������point��ֵ

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

	if (ifhd == 0)//���ǻҶ�ͼҪ����ת��
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	float x_rato_less = 1.0;
	float y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

   //����1��������
	cudaError_t t;
	refTex.addressMode[0] = cudaAddressModeBorder;
	refTex.addressMode[1] = cudaAddressModeBorder;
	refTex.normalized = false;
	refTex.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_move, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_move, cuArray_move);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_move, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	uchar* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

	dim3 block(8, 8);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	transformKernel_move << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less,x_move,y_move);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_move);
	cudaFree(pDstImgData1);
	//imshow("ԭͼ��", Lena);
	//imshow("�ƶ��Ժ��ͼ��", dstImg1);
	return dstImg1.clone();
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
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

//������˹����
	//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
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

//ͼ����б궨
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

	////ֻ����Ҷ�ֵ
	Mat Lena = image.clone();
	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int x_rato_less = 1;
	int y_rato_less = 1;

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_Laplace.addressMode[0] = cudaAddressModeBorder;
	refTex_Laplace.addressMode[1] = cudaAddressModeBorder;
	refTex_Laplace.normalized = false;
	refTex_Laplace.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_Laplace, &cuDesc_Laplace, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_Laplace, cuArray_Laplace);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_Laplace, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	Kerkel_Laplace <<<grid, block >>> (pDstImgData1, imgHeight_des_less, imgWidth_des_less ,mode ,c);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_Laplace);
	cudaFree(pDstImgData1);
	image=dstImg1.clone();
	//cout << image << endl;
	//cout<<image<<endl;
	//image.convertTo(image,CV_8U);
}