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
	
	cout<<"图像宽x："<<imgWidth_src<<endl;
	cout<<"图像高y：" <<imgHeight_src<<endl;

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;
		 
	int  BATCH = 1;
	int  NRANK = 2;

    cufftHandle plan;
	cufftComplex *data;
	
	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY*BATCH);

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