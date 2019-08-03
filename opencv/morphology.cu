#pragma once
#include "morphology.cuh"

texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_corrode;//用于计算双线性插值	

cudaArray* cuArray_corrode;//声明CUDA数组

//通道数
cudaChannelFormatDesc cuDesc_corrode = cudaCreateChannelDesc<uchar>();


//图像腐蚀change center
__device__ uchar change_center(int x, int y, Point_gpu* point_gpu,uchar* data,int len) {
	int x_N;
	int y_N;
	uchar result = 255;
	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (tex2D(refTex_corrode, x_N, y_N) < 255 && data[i]==255)//每个point_gpu位置上的像素都需要是255，否则该点将被腐蚀点
		{
			result = 0;
			break;
		}
	}
	return  result;
}

//图像腐蚀
__global__ void corrodeKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d,Point_gpu* point_gpu,uchar* data,int len)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			pDstImgData[idx] = (int)change_center(tidx,tidy,point_gpu,data,len);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


//图像膨胀change center
__device__ void change_expand(int* pDstImgData,int x, int y, int imgWidth_des_d, int imgHeight_des_d,Point_gpu* point_gpu,uchar* data, int len) {
	int x_N;
	int y_N;
	int idx;
	for (int i = 0; i < len; i++)
	{   
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (-1 < x_N && x_N < imgWidth_des_d && -1 < y_N && y_N < imgHeight_des_d)
		{   idx = (int)(y_N * imgWidth_des_d + x_N);
		
			 // printf("%u,%u \n", pDstImgData[idx], data[i]);
			if (pDstImgData[idx]==0 && tex2D(refTex_corrode, x_N, y_N)==255 && data[i] == 0)
			{
				int a =255;
				atomicExch(pDstImgData+idx,a);
				//printf("a: \n");
			   /* pDstImgData[idx] = data[i];*/
			}
			else {
				int a = (int)data[i];
				atomicExch(pDstImgData + idx, a);
				//pDstImgData[idx] =(int) data[i];
			}
		}
	}
}

//图像膨胀
__global__ void expandKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu,uchar* data,int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		if(tex2D(refTex_corrode,tidx,tidy)==255)//如何是255的像素检测是否需要膨胀
			change_expand(pDstImgData,tidx, tidy, imgWidth_des_d,imgHeight_des_d,point_gpu,data,len);//需要膨胀
		//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
	}
}

//mode=0 腐蚀，1=膨胀
Mat morphology_gpu(char * path,int len,Point_gpu*  point_offset_N,uchar* data ,int mode) {
	se_tpye * se_obj = (se_tpye*)malloc(sizeof(se_tpye));
	se_obj->init(len, point_offset_N,data);

	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena,1,"原图");
	
    int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//原图像宽
	int imgHeight_src = Lena.rows;//原图像高
	int channels = Lena.channels();


	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

	//设置1纹理属性
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//绑定cuArray到纹理
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//拷贝数据到cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//输出放缩以后在cpu上图像
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//缩小

	//输出放缩以后在cuda上的图像
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	if (mode==0)
	   corrodeKerkel << <grid, block >> > (pDstImgData1,imgHeight_des_less,imgWidth_des_less,se_obj->point_offset,se_obj->data,len);
	if (mode==1)
	   expandKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset,se_obj->data,len);
	cudaDeviceSynchronize();

	//从GPU拷贝输出数据到CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	stringstream ss;
	ss << len;
	string mark;
	ss >> mark;
	string ret = string("腐蚀以后的图") + mark;
	image_show(dstImg1,1,ret.c_str());
	//namedWindow("cuda_point最近插值：", WINDOW_NORMAL);
	//imshow("腐蚀以后的图像：", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}

//M是长，N是宽
Point_gpu* set_Point_gpu(int M, int N) {
	Point_gpu*  point_offset_N = (Point_gpu*)malloc(sizeof(Point_gpu) * M*N);
	int M_center = (int)M / 2;
	int N_center = (int)N / 2;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			point_offset_N[i*N + j].x = (int)j - N_center;
			point_offset_N[i*N + j].y = (int)i - M_center;
			//cout <<i<<"|"<<j<< endl;
			//cout<<point_offset_N[i*N + j].x <<endl;
			//cout<< point_offset_N[i*N + j].y << endl;
			//cout<<"--------------------------"<<endl;
		}
	}
	return point_offset_N;
}

//腐蚀用
uchar* set_Point_data(int M, int N) {
	uchar*  data = (uchar*)malloc(sizeof(uchar) * M*N);
	int M_center = (int)M / 2;
	int N_center = (int)N / 2;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			data[i*N + j]= 255;
		}
	}
	return data;
}

//膨胀用
uchar* set_Point_data_pz(int M, int N) {
	uchar*  data = (uchar*)malloc(sizeof(uchar)* M*N);
    data[0] = 0;
	data[1] = 255;
	data[2] = 0;
	data[3] = 255;
	data[4] = 255;
	data[5] = 255;
	data[6] = 0;
	data[7] = 255;
	data[8] = 0;
	return data;
}

void morphology_test(int M, int N,int mode)
{
	if(mode==0)
	{ Point_gpu* point_offset_N=set_Point_gpu(M,N);
	  uchar* data=set_Point_data(M,N);
	  morphology_gpu("C:/Users/Administrator/Desktop/opencv/m486.png", M*N, point_offset_N,data,0);
	}
	
	if(mode == 1)
	{ Point_gpu * point_offset_N =set_Point_gpu(M, N);;
	  //uchar* data = set_Point_data_pz(M,N);
	  uchar* data = set_Point_data(M, N);
	  morphology_gpu("C:/Users/Administrator/Desktop/opencv/font.png", M*N, point_offset_N, data, 1);
	}

}

//2、二值图像的逻辑集合运算,默认二值是由255和0构成
Mat AND_two(const Mat& A, const Mat& B,uchar min,uchar max) {//交函数
	Mat A_N =A.clone();
	A_N.convertTo(A_N, CV_8U);
	Mat B_N =B.clone();
	B_N.convertTo(B_N, CV_8U);
	Mat result = Mat::zeros(A_N.size(), CV_8U);

	int N = A.cols;
	int M = A.rows;
	for (size_t i = 0; i <M; i++)
		{  for (size_t j = 0; j <N; j++)
			   { 
				 if (A_N.at<uchar>(i, j) == max && B_N.at<uchar>(i, j) == max)
				 {
					 result.at<uchar>(i, j) = max;

				 }
				 else {
					 result.at<uchar>(i, j) = min;
				 }
			   }
		  }

	return result.clone();
}

Mat OR_two(const Mat& A, const Mat& B, uchar min, uchar max) {//并操作
	Mat A_N = A.clone();
	A_N.convertTo(A_N, CV_8U);
	Mat B_N = B.clone();
	B_N.convertTo(B_N, CV_8U);
	Mat result= Mat::zeros(A_N.size(), CV_8U);

	int N = A.cols;
	int M = A.rows;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (A_N.at<uchar>(i, j) == min && B_N.at<uchar>(i, j) == min)
			{
				result.at<uchar>(i, j) = min;

			}
			else {
				result.at<uchar>(i, j) = max;
			}
		}
	}

	return result.clone();
}

Mat NOT_two(const Mat& A, uchar min, uchar max) {//补操作
	Mat A_N = A.clone();
	A_N.convertTo(A_N, CV_8U);

	Mat result=Mat::zeros(A_N.size(),CV_8U);

	int N = A.cols;
	int M = A.rows;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (A_N.at<uchar>(i, j)== min)
			{
				result.at<uchar>(i, j) = max;

			}
			else {
				result.at<uchar>(i, j) = min;
			}
		}
	}

	return result.clone();
}

Mat AND_NOT_two(const Mat& A, const Mat& B, uchar min, uchar max) {//B补运算，再与A交
	Mat mide = NOT_two(B,0,255);
	image_show(mide, 1, "mide");
	Mat result = AND_two(A,mide,0,255);
	return result.clone();
}

Mat XOR_two(const Mat& A, const Mat& B, uchar min , uchar max) {//异或操作, 等于：A和B并集后剔除A交B
	Mat A_N = A.clone();
	A_N.convertTo(A_N, CV_8U);
	Mat B_N = B.clone();
	B_N.convertTo(B_N, CV_8U);
	Mat result;
	result.convertTo(result, CV_8U);

	int N = A.cols;
	int M = A.rows;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (A_N.at<uchar>(i, j)!=B_N.at<uchar>(i, j))
			{
				result.at<uchar>(i, j) = max;

			}
			else {
				result.at<uchar>(i, j) = min;
			}
		}
	}
	return result.clone();
}

//例9.5
void man_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/man.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 100, 255, 0);
	//image_show(Lena,1, "原图");
	
	int M = 5;
	int N = 5;
	Point_gpu* point=set_Point_gpu(M,N);
	uchar* data=set_Point_data(M, N);
	Mat mide=morphology_gpu("C:/Users/Administrator/Desktop/opencv/man.png", M*N,point, data, 0);
   
	Mat result=AND_NOT_two(Lena, mide,0,255);
	result.convertTo(result, CV_32F);
	image_show(result,1,"结果");
}

void chapter9() {
	man_test();
}