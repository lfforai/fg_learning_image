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
			if (tex2D(refTex_corrode, x_N, y_N) < 255 && data[i] == 255)//每个point_gpu位置上的像素都需要是255，否则该点将被腐蚀点
			{   //如果该点data【i】=0,表示不是腐蚀考虑条件
				result = 0;
				break;
			}
		}
		return  result;	
}

__device__ uchar change_center_catch(int x, int y, Point_gpu* point_gpu, uchar* data, int len) {
		int x_N;
		int y_N;
		uchar result;
		for (int i = 0; i < len; i++)
		{
			x_N = (int)(point_gpu[i].x + x);
			y_N = (int)(point_gpu[i].y + y);
			if (tex2D(refTex_corrode, x_N, y_N) == 255 && data[i] == 255)//每个point_gpu位置上的像素都需要是255，
			{   //如果该点data【i】=255,表示具备了可以填充的条件
				result = 255;
			}

			if (tex2D(refTex_corrode, x_N, y_N) == 0 && data[i] == 255)//如果不满足匹配条件，该0点还维持原0
			{   //如果该点像素与data【i】不一致,表示不是腐蚀考虑条件，该0点依然不填充	
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
			int index;
			
			if (data[(int)(len / 2)] == 255)//寻找255进行腐蚀测试
			{
				//printf("%u \n:", data[(int)(len / 2)]);
				if (tex2D(refTex_corrode, tidx, tidy) == 255)
					pDstImgData[idx] = (int)change_center(tidx, tidy, point_gpu, data, len);
			}
			else if (data[(int)(len / 2)] == 0) {
				if (tex2D(refTex_corrode, tidx, tidy) == 0)//寻找0进行击中击不中的填充
					pDstImgData[idx] = (int)change_center_catch(tidx, tidy, point_gpu, data, len);
			}
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


//图像膨胀change center
__device__ void change_expand(int* pDstImgData,int x, int y, int imgWidth_des_d, int imgHeight_des_d,Point_gpu* point_gpu,uchar* data, int len) {
	int x_N;
	int y_N;
	int idx;
	int a;
	for (int i = 0; i < len; i++)
	{   
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		a = (int)tex2D(refTex_corrode, x_N, y_N);
		if (~(y_N == 0 && x_N ==0) && -1 < x_N && x_N < imgWidth_des_d && -1 < y_N && y_N < imgHeight_des_d && data[i]==255 && a==0)
		{   idx = (int)(y_N * imgWidth_des_d + x_N);
		    a = 255;
			atomicExch(pDstImgData + idx, a);
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
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//缩小

	//输出放缩以后在cuda上的图像
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));
	t = cudaMemcpy(pDstImgData1, Lena.data, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	if (mode == 0)
		corrodeKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset, se_obj->data, len);
	if (mode == 1)
		expandKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset, se_obj->data, len);
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
	image_show(dstImg1, 1, ret.c_str());
	//namedWindow("cuda_point最近插值：", WINDOW_NORMAL);
	//imshow("腐蚀以后的图像：", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}

//mode=0 腐蚀，1=膨胀
Mat morphology_gpu_Mat(const Mat& image, int len, Point_gpu*  point_offset_N, uchar* data, int mode) {
	se_tpye * se_obj = (se_tpye*)malloc(sizeof(se_tpye));
	se_obj->init(len, point_offset_N, data);

	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);
	//image_show(Lena, 1, "原图");

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
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//缩小

	//输出放缩以后在cuda上的图像
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1,imgHeight_des_less * imgWidth_des_less * sizeof(int)); 
	t = cudaMemcpy(pDstImgData1,Lena.data,imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	if (mode == 0)
		corrodeKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset, se_obj->data, len);
	if (mode == 1)
		expandKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset, se_obj->data, len);
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
	image_show(dstImg1, 1, ret.c_str());
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

//击中、不中用
uchar* set_Point_data_jz(int M, int N) {
	uchar*  data = (uchar*)malloc(sizeof(uchar) * M*N);
	int M_center = (int)M / 2;
	int N_center = (int)N / 2;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			data[i*N + j] = 0;
		}
	}
	return data;
}

//非标准矩阵核，膨胀用
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

void test() {
	int M = 3;
	int N = 3;

	Mat X = Mat::zeros(Size(5, 5), CV_8U);
	X.at<uchar>(0, 1) = 255;
	X.at<uchar>(0, 0) = 255;
	X.at<uchar>(0, 2) = 255;
	X.at<uchar>(1, 1) = 255;
	//X.at<uchar>(2, 4) = 255;
	//X.at<uchar>(3, 3) = 255;
	cout << X << endl;
	Point_gpu* point = set_Point_gpu(M, N);
	uchar* data = set_Point_data_pz(M, N);
	Mat mide = morphology_gpu_Mat(X, M*N, point, data, 1);
	cout << mide << endl;
	cout << "-------------------------" << endl;
	for (size_t i = 0; i < 1; i++)
	{
		mide = morphology_gpu_Mat(mide, M*N, point, data, 1);
	}
	//mide.convertTo(mide, CV_8U);
	cout << mide << endl;
	cout << "-------------------------" << endl;
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
	  uchar* data = set_Point_data_pz(M,N);
	  //uchar* data = set_Point_data(M, N);
	  morphology_gpu("C:/Users/Administrator/Desktop/opencv/font1.png", M*N, point_offset_N, data, 1);
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
	uchar* data=set_Point_data(M,N);
	Mat mide=morphology_gpu("C:/Users/Administrator/Desktop/opencv/man.png", M*N,point, data, 0);
   
	Mat result=AND_NOT_two(Lena, mide,0,255);
	result.convertTo(result, CV_32F);
	image_show(result,1,"结果");
}

//例9.6
void remove_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/yuan.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena,1, "原图");

	//Mat X=Mat::zeros(Lena.size(),CV_8U);
	int M = 3;
	int N = 3;
	cv::Point p;//用image watch 查到的点
	p.x = 93;
	p.y = 49;
	Mat X = Mat::zeros(Lena.size(), CV_8U);
	X.at<uchar>(p.y,p.x)=255;

	Point_gpu* point = set_Point_gpu(M, N);
	uchar* data = set_Point_data_pz(M, N);
	//char* file = "C:/Users/Administrator/Desktop/opencv/yuan.png";
	Mat mide = morphology_gpu_Mat(X, M*N, point, data, 1);
	Mat result = AND_NOT_two(mide, Lena, 0, 255);

	for (size_t i = 0; i < 20; i++)
	{ 	mide = morphology_gpu_Mat(result, M*N, point, data, 1);
		result = AND_NOT_two(mide, Lena, 0, 255);
	}

	result=OR_two(result, Lena);
	result.convertTo(result, CV_32F);
	image_show(result, 1, "结果");
}

//统计一个二值图中的不为0的像素点个数
int cout_image_thread(Mat& image,int max=255)
{   Scalar ss=sum(image);
	return (int)(ss[0] / max);
}

//例9.7
void connection_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/chicken.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 190, 255, 0);
	image_show(Lena, 1, "原图");

	//由于只找一个区域，没有进行腐蚀操作
	//Mat X=Mat::zeros(Lena.size(),CV_8U);
	int M = 3;
	int N = 3;
	cv::Point p;//用image watch 查到的点
	p.x = 496;
	p.y = 89;
	Mat X = Mat::zeros(Lena.size(), CV_8U);
	X.at<uchar>(p.y, p.x) = 255;

	Point_gpu* point = set_Point_gpu(M, N);
	uchar* data = set_Point_data(M, N);//8连通
	//char* file = "C:/Users/Administrator/Desktop/opencv/yuan.png";
	Mat mide = morphology_gpu_Mat(X, M*N, point, data, 1);
	Mat result = AND_two(mide, Lena, 0, 255);

	int mark = 0;
	for (size_t i = 0; i <1000; i++)
	{
		mide = morphology_gpu_Mat(result, M*N, point, data, 1);
		result = AND_two(mide, Lena, 0, 255);
		if (mark == cout_image_thread(result, 255))
			break;
		else
			mark = cout_image_thread(result, 255);
	}

	cout << "连通分量中像素个数：" <<mark<< endl;
    //result = OR_two(result, Lena);
	result.convertTo(result, CV_32F);
	image_show(result, 1, "联通分量图");

}

//开集运算,先腐蚀，后膨胀,输入值已经是二值化以后的
Mat open_set(Mat& image,int M,int N,Point_gpu* point_N = NULL, uchar* data_N = NULL) {
	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);

	Point_gpu* point;
	uchar* data;
	if (NULL != point_N && NULL!= data_N)
	{
		point = point_N;
		data = data_N;
	}
	else {
		point = set_Point_gpu(M, N);
		data = set_Point_data(M, N);
	}
	
	Mat mide = morphology_gpu_Mat(Lena, M*N, point, data, 0);
	mide = morphology_gpu_Mat(mide, M*N, point, data, 1);
	return mide.clone();
}

//闭集运算,先膨胀，后腐蚀,输入值已经是二值化以后的
Mat close_set(Mat& image, int M, int N, Point_gpu* point_N=NULL, uchar* data_N=NULL) {
	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);

	Point_gpu* point;
	uchar* data;
	if (NULL != point_N && NULL != data_N)
	{  
		point = point_N;
		data = data_N;
	}
	else {
		point = set_Point_gpu(M, N);
		data = set_Point_data(M, N);
	}

	Mat mide = morphology_gpu_Mat(Lena, M*N, point, data, 1);
	mide = morphology_gpu_Mat(mide, M*N, point, data, 0);
	return mide.clone();
}

Mat bone_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/fy.jpg");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 100, 255, 0);
	Lena = NOT_two(Lena);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "原图");

	int M = 3;
	int N = 3;
	Point_gpu* point;
	uchar* data;

	point = set_Point_gpu(M, N);
	data = set_Point_data(M, N);
	Mat vector[500];
	Mat AkB = Lena;
	int k = 500;

	for (size_t i = 0; i < k; i++)
	{
		vector[i] = Mat::zeros(Lena.size(),CV_8U);
	}
	
	for (size_t i = 0; i < k; i++)
	{
		AkB = morphology_gpu_Mat(AkB, M*N, point, data, 0);
		if (i == 10)
		{
			Mat Akb_N = AkB.clone();
			Akb_N.convertTo(Akb_N, CV_32F);
			image_show(Akb_N, 1, "AKB10");
		}
		vector[i]=AND_NOT_two(AkB, open_set(AkB, M, N, point, data));
		if (i == 10)
		{
			Mat Akb_m = vector[i].clone();
			Akb_m.convertTo(Akb_m, CV_32F);
			image_show(Akb_m, 1, "AKB*B10");
		}
	}
	
	Mat result= vector[k-1];
	for (size_t i = 0; i < k-1; i++)
	{
		result=OR_two(result, vector[i]);
	}

	result.convertTo(result, CV_32F);
	image_show(result, 1, "结果");
	return result.clone();
}

//凸包计算测试
void Prot_shell() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/shell.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 100, 255, 0);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "原图");

	int M = 3;
	int N = 3;

	Point_gpu* point = set_Point_gpu(M, N);
	
	//B1
	uchar* B1 = set_Point_data_jz(M, N);
	B1[N * 0 + 0] = 255;
	B1[N * 1 + 0] = 255;
	B1[N * 2 + 0] = 255;

	//B2
	uchar* B2 = set_Point_data_jz(M, N);
	B2[N * 0 + 0] = 255;
	B2[N * 0 + 1] = 255;
	B2[N * 0 + 2] = 255;

	//B3
	uchar* B3 = set_Point_data_jz(M, N);
	B3[N * 0 + 2] = 255;
	B3[N * 1 + 2] = 255;
	B3[N * 2 + 2] = 255;

	//B4
	uchar* B4 = set_Point_data_jz(M, N);
	B4[N * 2 + 0] = 255;
	B4[N * 2 + 1] = 255;
	B4[N * 2 + 2] = 255;

	//B1 迭代
	Mat X1=Lena.clone();
	Mat X1_N=Lena.clone();
	while (true){
		X1 = OR_two(Lena,morphology_gpu_Mat(X1, M*N, point, B1, 0));
		if (cout_image_thread(X1) == cout_image_thread(X1_N))
			break;
		else
			X1_N = X1.clone();
	}

	//B2 迭代
	Mat X2 = Lena.clone();
	Mat X2_N = Lena.clone();
	while (true) {
		X2 = OR_two(Lena, morphology_gpu_Mat(X2, M*N, point, B2, 0));
		if (cout_image_thread(X2) == cout_image_thread(X2_N))
			break;
		else
			X2_N = X2.clone();
	}

	//B3 迭代
	Mat X3 = Lena.clone();
	Mat X3_N = Lena.clone();
	while (true) {
		X3 = OR_two(Lena, morphology_gpu_Mat(X3, M*N, point, B3, 0));
		if (cout_image_thread(X3) == cout_image_thread(X3_N))
			break;
		else
			X3_N = X3.clone();
	}

	//B4 迭代
	Mat X4 = Lena.clone();
	Mat X4_N = Lena.clone();
	while (true) {
		X4 = OR_two(Lena, morphology_gpu_Mat(X4, M*N, point, B4, 0));
		if (cout_image_thread(X4) == cout_image_thread(X4_N))
			break;
		else
			X4_N = X4.clone();
	}

	Mat result= OR_two(OR_two(OR_two(X1, X2), X3),X4);
	result.convertTo(result, CV_32F);
	image_show(result, 1, "结果");
}

//开集和闭集
void  open_close_test()
{
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/t.bmp");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	threshold(Lena, Lena, 10, 255, 0);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "原图");

	//由于只找一个区域，没有进行腐蚀操作
	int M = 100;
	int N = 100;

	Mat mide = open_set(Lena,M,N);
	mide.convertTo(mide, CV_32F);
	image_show(mide, 1, "开操作");

	Mat mide_close=close_set(Lena,M,N);
	mide_close.convertTo(mide_close, CV_32F);
	image_show(mide_close, 1, "闭操作");
}

void chapter9() {
	//test();//测试morphology_gpu是否正确
	//morphology_test(5, 5, 0);//例子9.1
	//morphology_test(3, 3, 1);//例子9.1
	//man_test();//例子9.5
	//remove_test();//例子9.6
	//connection_test();
	//open_close_test();
	//bone_test();
	Prot_shell();
}