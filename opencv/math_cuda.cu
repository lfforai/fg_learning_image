#pragma once
#include "math_cuda.cuh"

using namespace std;
using namespace cv;

//一些数学函数的测试
//一、四象限反正切函数atan2的输出结果
void  atan_cpu_test() {
	float pi = 3.1415926;
	printf("一象限,y=1.0,x=1.0，atan2:=%f度 \n", atan2(1.0, 1.0)*180.0/pi);
	printf("二象限,y=1.0,x=-1.0，atan2:=%f度 \n",atan2(1.0, -1.0)*180.0/pi);
	printf("三象限,y=-1.0,x=-1.0，atan2:=%f度 \n",atan2(-1.0, -1.0)*180.0/pi);
	printf("四象限,y=-1.0,x=1.0，atan2:=%f度 \n", atan2(-1.0, 1.0)*180.0/pi);
	printf("实验证明：4象限atan2返回值是弧度值.\n            返回值不是【0，2pi】而是【-pi，pi】之间\n");
}

//二、傅里叶变换F（-u，-v）= 
void cufft_math_test(char* path, int mode)
{   Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

	int imgWidth_src = Lena.cols;//原图像宽 x
	int imgHeight_src = Lena.rows;//原图像高 y

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;

	int  BATCH = 1;
	int  NRANK = 2;

	cufftHandle plan;
	cufftComplex *data;
	cufftComplex *data2;

	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY);//对f(x,y) 做 傅里叶变换
	cudaMallocManaged((void**)&data2, sizeof(cufftComplex)*NX*NY);//对f(x,y）做 反傅里叶变换

	//把图像元素赋值给赋值给实数部分
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = (float)Lena.data[NX*i + j];//非中心化
			data[NX*i + j].y = 0.0;

			data2[NX*i + j].x = (float)Lena.data[NX*i + j];//非中心化
			data2[NX*i + j].y = 0.0;
		}
	}

	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}

	/* Create a 2D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		NULL, 1, NX*NY, // *inembed, istride, idist
		NULL, 1, NX*NY, // *onembed, ostride, odist
		CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecC2C(plan, data2, data2, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}


	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	}

	for (int i = 0; i < 10; i++)
	{
		printf("%f,%f \n", data2[i].x, data2[i].y);

	}
	
	//测试对称性
	int not_zero_num = 0;
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{    
			if (abs(abs(data2[i*NX + j].x)*length - abs(data[(NY-1 - i)*NX + NX-1 - j].x)) > 0.00001 &&
				abs(abs(data2[i*NX + j].y)*length - abs(data[(NY-1 - i)*NX + NX-1 - j].y)) > 0.00001
				)
				not_zero_num++;
			//cout << abs(data2[i*NX + j].y - data[(NY - 1 - i)*NX + NX - 1 - j].y) << endl;
		}
	}

	cout<<"是否相等   ："<< not_zero_num <<endl;
	cufftDestroy(plan);
	cudaFree(data);
}

//图像傅里叶变换和反傅里叶变换还原
void hy_fun(Mat Lena_o) {
	Mat Lena = Lena_o.clone();

	int imgWidth_src = Lena.cols;//原图像宽 x
	int imgHeight_src = Lena.rows;//原图像高 y

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;

	int  BATCH = 1;
	int  NRANK = 2;

	cufftHandle plan;
	cufftComplex *data;

	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY);

	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}

	//把图像元素赋值给赋值给实数部分
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = (float)Lena.data[NX*i + j];//非中心化
			data[NX*i + j].y = 0.0;
		}
	}

	/* Create a 2D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		NULL, 1, NX*NY, // *inembed, istride, idist
		NULL, 1, NX*NY, // *onembed, ostride, odist
		CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}


	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
	}


	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}


	if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}

	//for (int i = 0; i < NY; i++)
	//{
	//	for (int j = 0; j < NX; j++)
	//	{
	//		cout <<"x:"<<data[NX*i + j].x<< endl;
	//	}
	//}

	Mat dstImg1 = Mat::zeros(NY, NX, CV_8UC1);//缩小
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			dstImg1.data[NX*i + j] = (uchar)(data[NX*i + j].x / length);
			//data[NX*i + j].y =(uchar)(data[NX*i + j].y / length);
		}
	}

	cufftDestroy(plan);
	imshow("cufft原图:", dstImg1);
}