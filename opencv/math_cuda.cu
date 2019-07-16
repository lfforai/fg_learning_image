#pragma once
#include "math_cuda.cuh"

using namespace std;
using namespace cv;

//һЩ��ѧ�����Ĳ���
//һ�������޷����к���atan2��������
void  atan_cpu_test() {
	float pi = 3.1415926;
	printf("һ����,y=1.0,x=1.0��atan2:=%f�� \n", atan2(1.0, 1.0)*180.0/pi);
	printf("������,y=1.0,x=-1.0��atan2:=%f�� \n",atan2(1.0, -1.0)*180.0/pi);
	printf("������,y=-1.0,x=-1.0��atan2:=%f�� \n",atan2(-1.0, -1.0)*180.0/pi);
	printf("������,y=-1.0,x=1.0��atan2:=%f�� \n", atan2(-1.0, 1.0)*180.0/pi);
	printf("ʵ��֤����4����atan2����ֵ�ǻ���ֵ.\n            ����ֵ���ǡ�0��2pi�����ǡ�-pi��pi��֮��\n");
}

//��������Ҷ�任F��-u��-v��= 
void cufft_math_test(char* path, int mode)
{   Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	int imgWidth_src = Lena.cols;//ԭͼ��� x
	int imgHeight_src = Lena.rows;//ԭͼ��� y

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;

	int  BATCH = 1;
	int  NRANK = 2;

	cufftHandle plan;
	cufftComplex *data;
	cufftComplex *data2;

	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY);//��f(x,y) �� ����Ҷ�任
	cudaMallocManaged((void**)&data2, sizeof(cufftComplex)*NX*NY);//��f(x,y���� ������Ҷ�任

	//��ͼ��Ԫ�ظ�ֵ����ֵ��ʵ������
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = (float)Lena.data[NX*i + j];//�����Ļ�
			data[NX*i + j].y = 0.0;

			data2[NX*i + j].x = (float)Lena.data[NX*i + j];//�����Ļ�
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
	
	//���ԶԳ���
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

	cout<<"�Ƿ����   ��"<< not_zero_num <<endl;
	cufftDestroy(plan);
	cudaFree(data);
}

//ͼ����Ҷ�任�ͷ�����Ҷ�任��ԭ
void hy_fun(Mat Lena_o) {
	Mat Lena = Lena_o.clone();

	int imgWidth_src = Lena.cols;//ԭͼ��� x
	int imgHeight_src = Lena.rows;//ԭͼ��� y

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

	//��ͼ��Ԫ�ظ�ֵ����ֵ��ʵ������
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = (float)Lena.data[NX*i + j];//�����Ļ�
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

	Mat dstImg1 = Mat::zeros(NY, NX, CV_8UC1);//��С
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			dstImg1.data[NX*i + j] = (uchar)(data[NX*i + j].x / length);
			//data[NX*i + j].y =(uchar)(data[NX*i + j].y / length);
		}
	}

	cufftDestroy(plan);
	imshow("cufftԭͼ:", dstImg1);
}