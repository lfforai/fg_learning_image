#pragma once
#include "cufft.cuh"

//利用cuda自带cufft模块 实现图像的傅里叶变换
using namespace std;
using namespace cv;

//mode=0普通频谱，mode=1中心化频谱，mode=2中心化后对数化
void cufft(char* path,int mode) {
	Mat Lena = imread(path);
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
	
	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY);

	//把图像元素赋值给赋值给实数部分
	for (int i = 0;i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			if (mode == 1 || mode == 2)
				data[NX*i + j].x = (float)Lena.data[NX*i + j] * pow(-1.0, i + j);//(0,0)转移到（N/2,N/2)中心
			else
				data[NX*i + j].x = (float)Lena.data[NX*i + j] ;//非中心化
			data[NX*i + j].y = 0.0;
	/*	    if(i==0 && j==0)
			  printf("aa:%f，%f \n", data[NX*i + j].x, data[NX*i + j].y);*/
		}
	}

	cout<<"--------------原图前10个像素:---------------"<<endl;
	for (int i = 0; i < 10; i++) {
		cout << "i:="<<i<< "，实部:" << data[i].x << "|虚部:" << data[i].y << endl;
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

	cout<<"--------------傅里叶变换:--------------------"<<endl;
	for (int i = 0; i < 10; i++) {
		cout << "i:=" << i << "，实部:" << data[i].x << "|虚部:" << data[i].y << endl;
	}
	
	//绘制频谱图，无中心化、中心化、对数化 //这部分完全可以cuda化，目前没有做这项工作
	cout << "--------------绘制频谱图:--------------------" << endl;
	float* data_spectrum;
	cudaMallocManaged((void**)&data_spectrum, sizeof(float)*NX*NY);
	
	uchar* data_spectrum_uchar;
	cudaMallocManaged((void**)&data_spectrum_uchar, sizeof(uchar)*NX*NY);

	//计算频谱
	float max = 0.0f;
	float min = 10000000000000.0f;
	for (int i = 0; i < NY; i++)
	{   
		for (int j = 0; j < NX; j++)
		{   if (mode==1 || mode==0)
			   data_spectrum[NX*i+j]=sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2));
		    if (mode==2)
		       data_spectrum[NX*i+j] =1+log(sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2)));//对数
			if (j == 0 && i == 0) {
				min = data_spectrum[NX*i + j];
				max = data_spectrum[NX*i + j];
			}
			else {
				if (data_spectrum[NX*i + j] < min)
					min = data_spectrum[NX*i + j];
				if (data_spectrum[NX*i + j] > max)
					max = data_spectrum[NX*i + j];
			}
		}
	}

	//归一化以后，把频率变为图像格式
	float max_min = max- min;
	cout<<max_min<<endl;
	cout<<"最大值："<<max<< endl;
	cout<<"最小值："<<min<< endl;
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data_spectrum_uchar[NX*i + j]=(uchar)(((data_spectrum[NX*i + j] - min)/ max_min)*255);
		}
	}

	Mat dstImg1 = Mat::zeros(NY,NX, CV_8UC1);//缩小
	cudaMemcpy(dstImg1.data, data_spectrum_uchar, NX * NY * sizeof(uchar), cudaMemcpyDefault);
	if(mode==0)
	  imshow("原始频谱图：", dstImg1);
	if(mode==1)
	  imshow("中心化频谱图：", dstImg1);
	if(mode==2)
	  imshow("中心及对数化频谱图：", dstImg1);

	cout << "--------------傅里叶反变换:--------------------" << endl;
	if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}
	
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	}

	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = data[NX*i + j].x / length;
			data[NX*i + j].y = data[NX*i + j].y / length;
		}
	}

	for (int i = 0; i < 10; i++) {
		cout << "i:=" << i << "，实部:" << data[i].x << "|虚部:" << data[i].y << endl;
	}

	cufftDestroy(plan);
	cudaFree(data);
}

void cuffttest(char *path){
	//char *path = "C:/Users/Administrator/Desktop/I.png";
	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	imshow("原图：", Lena);
	cufft(path, 0);//原始频率图
	cufft(path, 1);//中心化
	cufft(path, 2);//对数化
	waitKey(0);
}

//通用傅里叶包
//mode=0:傅里叶正变换,mode=1:傅里叶逆变换
//m_mode=0：普通化，m_mode=1，中心化
//如果是Ｆ（ｕ，ｖ）转换为ｆ（ｘ，ｙ），需要对反傅里叶*1/MN，MN=1为需要,ifgray是否需要灰度化
cufftComplex* cufft_fun(const char* path,Mat Lena_o,int mode,int m_mode,int MN,int ifgray) {
	Mat Lena;
	if (strlen(path)== 0)
	   Lena=Lena_o.clone();
	else
	   Lena = imread(path);

	if (ifgray==1)
	   cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

	int imgWidth_src = Lena.cols;//原图像宽 x
	int imgHeight_src = Lena.rows;//原图像高 y

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;
	cout<<NX<<","<<NY<<endl;

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
			if (mode == 0  && m_mode == 1)//傅里叶正变换和中心化
				 data[NX*i + j].x = (float)Lena.data[NX*i + j] * pow(-1.0, i + j);//(0,0)转移到（N/2,N/2)中心
			else if(mode==0 && m_mode==0)
			     data[NX*i + j].x = (float)Lena.data[NX*i + j];//非中心化
			else if(mode==1)
				 data[NX*i + j].x = (float)Lena.data[NX*i + j];//非中心化
			data[NX*i + j].y = 0.0;
			/*if(i==0 && j==0)
			  printf("aa:%f，%f \n", data[NX*i + j].x, data[NX*i + j].y);*/
		}
	}

	/* Create a 2D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		NULL, 1, NX*NY, // *inembed, istride, idist
		NULL, 1, NX*NY, // *onembed, ostride, odist
		CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}


	if (mode == 0) //傅里叶变换
	{   /* Use the CUFFT plan to transform the signal in place. */
		if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		}

		if (cudaDeviceSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	else 
	{   //逆傅里叶变换
		if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		}

		if (cudaDeviceSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		if (MN==1)
		{
			for (int i = 0; i < NY; i++)
			{
				for (int j = 0; j < NX; j++)
				{
					data[NX*i + j].x = data[NX*i + j].x / length;
					data[NX*i + j].y = data[NX*i + j].y / length;
				}
			}
		}
	}

	cufftDestroy(plan);
	return data;
}


//输入是已经傅里叶或者反傅里叶化的cufftComplex类型，输出可以绘制图像的频谱Mat图
//mode=0：普通频谱，mode=1：log频谱转换
Mat fre_spectrum(cufftComplex* data,int NX,int NY,int mode) {

	//绘制频谱图，无中心化、中心化、对数化 //这部分完全可以cuda化，目前没有做这项工作
	float* data_spectrum;
	cudaMallocManaged((void**)&data_spectrum, sizeof(float)*NX*NY);

	uchar* data_spectrum_uchar;
	cudaMallocManaged((void**)&data_spectrum_uchar, sizeof(uchar)*NX*NY);

	//计算频谱
	float max = 0.0f;
	float min = 10000000000000.0f;
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			if (mode == 0)
				data_spectrum[NX*i + j] = sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2));
			if (mode == 1)
				data_spectrum[NX*i + j] = log(1+sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2)));//对数
			if (j == 0 && i == 0) {
				min = data_spectrum[NX*i + j];
				max = data_spectrum[NX*i + j];
			}
			else {
				if (data_spectrum[NX*i + j] < min)
					min = data_spectrum[NX*i + j];
				if (data_spectrum[NX*i + j] > max)
					max = data_spectrum[NX*i + j];
			}
		}
	}

	//归一化以后，把频率变为图像格式
	float max_min = max - min;

	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data_spectrum_uchar[NX*i + j] = (uchar)(((data_spectrum[NX*i + j] - min) / max_min) * 255);
		}
	}

	Mat dstImg1 = Mat::zeros(NY, NX, CV_8UC1);//缩小
	cudaMemcpy(dstImg1.data, data_spectrum_uchar, NX * NY * sizeof(uchar), cudaMemcpyDefault);
	cudaFree(data_spectrum);
	cudaFree(data_spectrum_uchar);
	return dstImg1;
}

//输入是已经傅里叶或者反傅里叶化的cufftComplex类型，输出可以绘制图像的相位Mat图
//mode=0：普通频谱，mode=1：log频谱转换
Mat angle_spectrum(cufftComplex* data, int NX, int NY) {

	//绘制频谱图，无中心化、中心化、对数化 //这部分完全可以cuda化，目前没有做这项工作
	float* data_spectrum1;
	cudaMallocManaged((void**)&data_spectrum1, sizeof(float)*NX*NY);

	uchar* data_spectrum_uchar1;
	cudaMallocManaged((void**)&data_spectrum_uchar1, sizeof(uchar)*NX*NY);

	//计算频谱
	float max = 0.0f;
	float min = 10000000000000.0f;
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{   
            //float temp=atan2(data[NX*i + j].y,data[NX*i + j].x)*180.0/3.1415926;
            //if(temp<0)
            //   temp=360.0+temp;
			float temp = atan2(data[NX*i + j].y, data[NX*i + j].x);
			if (temp < 0)
				temp = 2*3.1415926 + temp;
            data_spectrum1[NX*i + j] = log(1+temp);
			//cout<<data_spectrum1[NX*i + j]<<endl;
			if (j == 0 && i == 0) {
				min = data_spectrum1[NX*i + j];
				max = data_spectrum1[NX*i + j];
			}
			else {
				if (data_spectrum1[NX*i + j] < min)
					min = data_spectrum1[NX*i + j];
				if (data_spectrum1[NX*i + j] > max)
					max = data_spectrum1[NX*i + j];
			}
		}
	}

	//归一化以后，把频率变为图像格式
	float max_min = max - min;

	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data_spectrum_uchar1[NX*i + j] = (uchar)(((data_spectrum1[NX*i + j] - min) / max_min) * 255);
		}
	}

	Mat dstImg1 = Mat::zeros(NY, NX, CV_8UC1);//缩小
	cudaMemcpy(dstImg1.data, data_spectrum_uchar1, NX * NY * sizeof(uchar), cudaMemcpyDefault);
	cudaFree(data_spectrum1);
	cudaFree(data_spectrum_uchar1);
	return dstImg1;
}