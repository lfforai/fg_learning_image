#pragma once
#include "opencv_chapter9.cuh"
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_corrode;//���ڼ���˫���Բ�ֵ

cudaArray* cuArray_corrode;//����CUDA����

//ͨ����
cudaChannelFormatDesc cuDesc_corrode = cudaCreateChannelDesc<uchar>();

//ͼ��ʴchange center
__device__ uchar change_center(int x, int y, Point_gpu* point_gpu,uchar* data,int len) {
	
		int x_N;
		int y_N;
		uchar result = 255;
		for (int i = 0; i < len; i++)
		{
			x_N = (int)(point_gpu[i].x + x);
			y_N = (int)(point_gpu[i].y + y);
			if (tex2D(refTex_corrode, x_N, y_N) < 255 && data[i] == 255)//ÿ��point_gpuλ���ϵ����ض���Ҫ��255������õ㽫����ʴ��
			{   //����õ�data��i��=0,��ʾ���Ǹ�ʴ��������
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
			if (tex2D(refTex_corrode, x_N, y_N) == 255 && data[i] == 255)//ÿ��point_gpuλ���ϵ����ض���Ҫ��255��
			{   //����õ�data��i��=255,��ʾ�߱��˿�����������
				result = 255;
			}

			if (tex2D(refTex_corrode, x_N, y_N) == 0 && data[i] == 255)//���������ƥ����������0�㻹ά��ԭ0
			{   //����õ�������data��i����һ��,��ʾ���Ǹ�ʴ������������0����Ȼ�����	
		    	result = 0;
				break;
			}
		}
		return  result;
}

//ͼ��ʴ
__global__ void corrodeKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d,Point_gpu* point_gpu,uchar* data,int len)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			int index;
			
			if (data[(int)(len / 2)] == 255)//Ѱ��255���и�ʴ����
			{
				//printf("%u \n:", data[(int)(len / 2)]);
				if (tex2D(refTex_corrode, tidx, tidy) == 255)
					pDstImgData[idx] = (int)change_center(tidx, tidy, point_gpu, data, len);
			}
			else if (data[(int)(len / 2)] == 0) {
				if (tex2D(refTex_corrode, tidx, tidy) == 0)//Ѱ��0���л��л����е����
					pDstImgData[idx] = (int)change_center_catch(tidx, tidy, point_gpu, data, len);
			}
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


//ͼ������change center
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

//ͼ������
__global__ void expandKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu,uchar* data,int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		if(tex2D(refTex_corrode,tidx,tidy)==255)//�����255�����ؼ���Ƿ���Ҫ����
			change_expand(pDstImgData,tidx, tidy, imgWidth_des_d,imgHeight_des_d,point_gpu,data,len);//��Ҫ����
		//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
	}
}



//mode=0 ��ʴ��1=����
Mat morphology_gpu(char * path,int len,Point_gpu*  point_offset_N,uchar* data ,int mode) {
	se_tpye * se_obj = (se_tpye*)malloc(sizeof(se_tpye));
	se_obj->init(len, point_offset_N,data);

	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena,1,"ԭͼ");
	
	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
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

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	stringstream ss;
	ss << len;
	string mark;
	ss >> mark;
	string ret = string("��ʴ�Ժ��ͼ") + mark;
	image_show(dstImg1, 1, ret.c_str());
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}

//mode=0 ��ʴ��1=����
Mat morphology_gpu_Mat(const Mat& image, int len, Point_gpu*  point_offset_N, uchar* data, int mode) {
	se_tpye * se_obj = (se_tpye*)malloc(sizeof(se_tpye));
	se_obj->init(len, point_offset_N, data);

	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);
	//image_show(Lena, 1, "ԭͼ");

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
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

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	stringstream ss;
	ss << len;
	string mark;
	ss >> mark;
	string ret = string("��ʴ�Ժ��ͼ") + mark;
	image_show(dstImg1, 1, ret.c_str());
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}


//M�ǳ���N�ǿ�
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


//M�ǳ���N�ǿ�,����һ���ڽ�Բ��-ƽ̹�ṹԪ
int* set_Point_data_circle(int M) 
{
	//�����ڲ�����
	auto ifin = [](int x_rows, int y_cols, int P_mide, int  Q_mide, int D0_radius)->bool {
		bool r = false;//������D0��
		if ((float)sqrt(pow(x_rows - P_mide, 2.0) + pow(y_cols - Q_mide, 2.0)) <=(float) D0_radius)
		{
			/*cout<< (float)sqrt(pow(x_rows - P_mide, 2.0) + pow(y_cols - Q_mide, 2.0)) <<endl;*/
			r = true;
		}
		return r;
	};

	int*  data = (int*)malloc(sizeof(int) * M*M);
	int M_center = (int)M / 2;
	int N_center = (int)M / 2;
	cout<< M_center <<endl;
	for(size_t i = 0; i < M; i++)
	{   for(size_t j = 0; j < M; j++)
		  {   if(ifin(j, i, M_center, N_center, M_center))
			    {data[i*M + j] =-255;//���ܺ��Ե�
			    }else{
				 data[i*M + j] =-1;//���Ժ��Ե�
                }
		  }
	}
	return data;
}


//��ʴ��
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

//���С�������
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

//�Ǳ�׼����ˣ�������
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

//2����ֵͼ����߼���������,Ĭ�϶�ֵ����255��0����
Mat AND_two(const Mat& A, const Mat& B,uchar min,uchar max) {//������
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

Mat OR_two(const Mat& A, const Mat& B, uchar min, uchar max) {//������
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

Mat NOT_two(const Mat& A, uchar min, uchar max) {//������
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

Mat AND_NOT_two(const Mat& A, const Mat& B, uchar min, uchar max) {//B�����㣬����A��
	Mat mide = NOT_two(B,0,255);
	image_show(mide, 1, "mide");
	Mat result = AND_two(A,mide,0,255);
	return result.clone();
}

Mat XOR_two(const Mat& A, const Mat& B, uchar min , uchar max) {//������, ���ڣ�A��B�������޳�A��B
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

//��9.5
void man_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/man.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	//image_show(Lena,1, "ԭͼ");
	
	int M = 5;
	int N = 5;
	Point_gpu* point=set_Point_gpu(M,N);
	uchar* data=set_Point_data(M,N);
	Mat mide=morphology_gpu("C:/Users/Administrator/Desktop/opencv/man.png", M*N,point, data, 0);
   
	Mat result=AND_NOT_two(Lena, mide,0,255);
	result.convertTo(result, CV_32F);
	image_show(result,1,"���");
}

//��9.6
void remove_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/yuan.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena,1, "ԭͼ");

	//Mat X=Mat::zeros(Lena.size(),CV_8U);
	int M = 3;
	int N = 3;
	cv::Point p;//��image watch �鵽�ĵ�
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
	image_show(result, 1, "���");
}

//ͳ��һ����ֵͼ�еĲ�Ϊ0�����ص����
int cout_image_thread(Mat& image,int max=255)
{   Scalar ss=sum(image);
	return (int)(ss[0] / max);
}

//��9.7
void connection_test() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/chicken.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 190, 255, 0);
	image_show(Lena, 1, "ԭͼ");

	//����ֻ��һ������û�н��и�ʴ����
	//Mat X=Mat::zeros(Lena.size(),CV_8U);
	int M = 3;
	int N = 3;
	cv::Point p;//��image watch �鵽�ĵ�
	p.x = 496;
	p.y = 89;
	Mat X = Mat::zeros(Lena.size(), CV_8U);
	X.at<uchar>(p.y, p.x) = 255;

	Point_gpu* point = set_Point_gpu(M, N);
	uchar* data = set_Point_data(M, N);//8��ͨ
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

	cout << "��ͨ���������ظ�����" <<mark<< endl;
    //result = OR_two(result, Lena);
	result.convertTo(result, CV_32F);
	image_show(result, 1, "��ͨ����ͼ");

}

//��������,�ȸ�ʴ��������,����ֵ�Ѿ��Ƕ�ֵ���Ժ��
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

//�ռ�����,�����ͣ���ʴ,����ֵ�Ѿ��Ƕ�ֵ���Ժ��
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
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	Lena = NOT_two(Lena);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "ԭͼ");

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
	image_show(result, 1, "���");
	return result.clone();
}

//͹���������
void Prot_shell() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/shell.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "ԭͼ");

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

	//B1 ����
	Mat X1=Lena.clone();
	Mat X1_N=Lena.clone();
	while (true){
		X1 = OR_two(Lena,morphology_gpu_Mat(X1, M*N, point, B1, 0));
		if (cout_image_thread(X1) == cout_image_thread(X1_N))
			break;
		else
			X1_N = X1.clone();
	}

	//B2 ����
	Mat X2 = Lena.clone();
	Mat X2_N = Lena.clone();
	while (true) {
		X2 = OR_two(Lena, morphology_gpu_Mat(X2, M*N, point, B2, 0));
		if (cout_image_thread(X2) == cout_image_thread(X2_N))
			break;
		else
			X2_N = X2.clone();
	}

	//B3 ����
	Mat X3 = Lena.clone();
	Mat X3_N = Lena.clone();
	while (true) {
		X3 = OR_two(Lena, morphology_gpu_Mat(X3, M*N, point, B3, 0));
		if (cout_image_thread(X3) == cout_image_thread(X3_N))
			break;
		else
			X3_N = X3.clone();
	}

	//B4 ����
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
	image_show(result, 1, "���");
}

//�����ͱռ�
void  open_close_test()
{
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/t.bmp");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 10, 255, 0);
	//Mat Lena_close=Lena.clone();
	image_show(Lena, 1, "ԭͼ");

	//����ֻ��һ������û�н��и�ʴ����
	int M = 100;
	int N = 100;

	Mat mide = open_set(Lena,M,N);
	mide.convertTo(mide, CV_32F);
	image_show(mide, 1, "������");

	Mat mide_close=close_set(Lena,M,N);
	mide_close.convertTo(mide_close, CV_32F);
	image_show(mide_close, 1, "�ղ���");
}


//--------------------------------------�Ҷ���̬ͼ��ѧ------------------------------------
//�Ҷ�ͼ��ʴ-ƽ̹Բ
__device__ int change_center_gray(int x, int y,int imgHeight_des_d,int imgWidth_des_d,Point_gpu* point_gpu, int* data, int len) {
	int x_N;
	int y_N;
	int min=1000;

	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (data[i] == -255 && -1 < x_N && x_N < imgWidth_des_d && -1 < y_N && y_N < imgHeight_des_d)
		{   //����õ�data��i��=0,��ʾ���Ǹ�ʴ��������
			if ((int)tex2D(refTex_corrode, x_N, y_N) < min)
			   min = (int)tex2D(refTex_corrode, x_N, y_N);
		}
	}
	return  min;
}


//�Ҷ�ͼ��ʴ
__global__ void corrodeKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu, int* data, int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
        pDstImgData[idx] = change_center_gray(tidx, tidy,imgHeight_des_d,imgWidth_des_d,point_gpu, data, len);
	}
}

//�Ҷ�ͼ������
__device__ int change_expand_gray(int x, int y, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu, int* data, int len) {
	int x_N;
	int y_N;
	int max = -265;

	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (data[i] == -255 && -1 < x_N && x_N < imgWidth_des_d && -1 < y_N && y_N < imgHeight_des_d)
		{   //����õ�data��i��=0,��ʾ���Ǹ�ʴ��������
			if ((int)tex2D(refTex_corrode, x_N, y_N) > max)
				max = (int)tex2D(refTex_corrode, x_N, y_N);
		}
	}
	return  max;
}

//�Ҷ�ͼ������
__global__ void expandKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu, int* data, int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
        pDstImgData[idx] = change_expand_gray(tidx, tidy, imgHeight_des_d, imgWidth_des_d, point_gpu, data, len);
	}
}

//�Ҷ�ͼ��ʴ������
Mat morphology_gpu_gray(char * path, int len, Point_gpu*  point_offset_N, int* data, int mode) {
	se_tpye_gray * se_obj = (se_tpye_gray*)malloc(sizeof(se_tpye_gray));
	se_obj->init(len, point_offset_N, data);

	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena, 1, "ԭͼ");

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
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

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	stringstream ss;
	ss << len;
	string mark;
	ss >> mark;
	string ret;
	if (mode==1)
	   ret= string("����")+string("��ͼ") + mark;
	else
	   ret = string("��ʴ") + string("��ͼ") + mark;
	image_show(dstImg1, 1, ret.c_str());
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}

//�Ҷ�ͼ��ʴ������
Mat morphology_gpu_gray_Mat(Mat& Lena_N, int len, Point_gpu*  point_offset_N, int* data, int mode) {
	se_tpye_gray * se_obj = (se_tpye_gray*)malloc(sizeof(se_tpye_gray));
	se_obj->init(len, point_offset_N, data);

	Mat Lena = Lena_N.clone();
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//threshold(Lena, Lena, 100, 255, 0);
	//image_show(Lena, 1, "ԭͼ");

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
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

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	//stringstream ss;
	//ss << len;
	//string mark;
	//ss >> mark;
	//string ret;
	//if (mode == 1)
	//	ret = string("����") + string("��ͼ") + mark;
	//else
	//	ret = string("��ʴ") + string("��ͼ") + mark;
	//image_show(dstImg1, 1, ret.c_str());
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	return dstImg1.clone();
}

Mat open_set_gray(Mat& image, int M, int N, Point_gpu* point_N = NULL, int* data_N = NULL) {
	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);

	Point_gpu* point;
	int* data;
	if (NULL != point_N && NULL != data_N)
	{
		point = point_N;
		data = data_N;
	}
	else {
		point = set_Point_gpu(M, N);
		data = set_Point_data_circle(M);
	}

	Mat mide = morphology_gpu_gray_Mat(Lena, M*N, point, data, 0);
	mide.convertTo(mide, CV_8U);
	mide = morphology_gpu_gray_Mat(mide, M*N, point, data, 1);
	return mide.clone();
}

//�ռ�����,�����ͣ���ʴ,����ֵ�Ѿ��Ƕ�ֵ���Ժ��
Mat close_set_gray(Mat& image, int M, int N, Point_gpu* point_N = NULL, int* data_N = NULL) {
	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);

	Point_gpu* point;
	int* data;
	if (NULL != point_N && NULL != data_N)
	{
		point = point_N;
		data = data_N;
	}
	else {

		point = set_Point_gpu(M, N);
		data = set_Point_data_circle(M);
	}

	Mat mide = morphology_gpu_gray_Mat(Lena, M*N, point, data, 1);
	mide.convertTo(mide, CV_8U);
	mide = morphology_gpu_gray_Mat(mide, M*N, point, data, 0);
	return mide.clone();
}

///�Ҷ���̬ͼ��ѧ ����
void gray_test() {

	int M = 3;
	int N = 3;
	Point_gpu* point = set_Point_gpu(M, N);
	int* data = set_Point_data_circle(M);

	//morphology_gpu_gray("C:/Users/Administrator/Desktop/opencv/dl.png", M*N, point, data, 0);
	//morphology_gpu_gray("C:/Users/Administrator/Desktop/opencv/dl.png", M*N, point, data, 1);
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/dl.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	image_show(Lena,1,"ԭͼ");
	Mat close=close_set_gray(Lena, M, N, NULL, NULL);
	image_show(close, 1,"close");
	Mat open=open_set_gray(Lena, 10, 10, NULL, NULL);
	image_show(open, 1,"open");
}

void chapter9() {
	//test();//����morphology_gpu�Ƿ���ȷ
	//morphology_test(5, 5, 0);//����9.1
	//morphology_test(3, 3, 1);//����9.1
	//man_test();//����9.5
	//remove_test();//����9.6
	//connection_test();
	//open_close_test();
	//bone_test();
	//Prot_shell();
	gray_test();
}


//---------------------------------------------��ʮ��----------------------------------
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_space_filter;//���ڼ���˫���Բ�ֵ

cudaArray* cuArray_space_filter;//����CUDA����

//ͨ����
cudaChannelFormatDesc cuDesc_space_filter = cudaCreateChannelDesc<uchar>();//ͨ����

////�ռ��˲�
template<class T>
__device__ int spacefilter(int x, int y, Point_gpu* point_gpu, T* data, int len) {
	int x_N;
	int y_N;
	float result = 0;
	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		result = result + (((float)tex2D(refTex_space_filter, x_N, y_N))*((float)data[i]));
	    //if (x == 0 && y == 0)
				//printf("x:%d,y:%d,|%d,%f,%d,%f \n",x_N,y_N,i, (((float)tex2D(refTex_space_filter, x_N, y_N))*((float)data[i])),(int)tex2D(refTex_space_filter, x_N, y_N),data[i]);
	}
	return  (int)result;
}

////�ռ��˲�
template<class T>
__global__ void space_filter_Kerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu, T* data, int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	//if (tidx == 0 && tidy == 0) {
	//	printf("%f \n", atan2(-1.0, 1.0)/(2.0*3.1415926)*360.0);
	//}

	if(tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	  {
		int idx = tidy * imgWidth_des_d + tidx;
		pDstImgData[idx] = spacefilter<T>(tidx, tidy, point_gpu, data, len);
	  }
}

template<class T>
Mat space_filter_cpu(char * path, int len, Point_gpu*  point_offset_N, T* data, float size)
{
	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	image_show(Lena, size, "ԭͼ");

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_space_filter.addressMode[0] = cudaAddressModeClamp;
	refTex_space_filter.addressMode[1] = cudaAddressModeClamp;
	refTex_space_filter.normalized = false;
	refTex_space_filter.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_space_filter, &cuDesc_space_filter, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_space_filter, cuArray_space_filter);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_space_filter, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	//cout<<Lena<<endl;
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));
	t = cudaMemcpy(pDstImgData1, Lena.data, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	space_filter_Kerkel<T> << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, point_offset_N, data, len);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_space_filter);
	cudaFree(pDstImgData1);
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	//Lena = Lena - dstImg1;
	//Lena.convertTo(Lena, CV_8U);
	//imshow("���򱱼�_������˹�任���ͼ", Lena);
	//result.convertTo(result, CV_32F);
	//image_show(Lena, 1, "�仯���ͼ");
	return dstImg1.clone();
}

template<class T>
Mat space_filter_cpu_mat(Mat image, int len, Point_gpu*  point_offset_N, T* data, float size)
{
	Mat Lena = image.clone();
	Lena.convertTo(Lena, CV_8U);

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_space_filter.addressMode[0] = cudaAddressModeClamp;
	refTex_space_filter.addressMode[1] = cudaAddressModeClamp;
	refTex_space_filter.normalized = false;
	refTex_space_filter.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_space_filter, &cuDesc_space_filter, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_space_filter, cuArray_space_filter);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_space_filter, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Lena.convertTo(Lena, CV_32S);
	//cout<<Lena<<endl;
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));
	t = cudaMemcpy(pDstImgData1, Lena.data, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	space_filter_Kerkel<T> << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, point_offset_N, data, len);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_space_filter);
	cudaFree(pDstImgData1);
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
	//Lena = Lena - dstImg1;
	//Lena.convertTo(Lena, CV_8U);
	//imshow("���򱱼�_������˹�任���ͼ", Lena);
	//result.convertTo(result, CV_32F);
	//image_show(Lena, 1, "�仯���ͼ");
	return dstImg1.clone();
}

//��������
void single_point()
{
	int M = 3;
	int N = 3;
	filter_screem<int>* filter = (filter_screem<int>*)malloc(sizeof(filter_screem<int>));
	filter->init(M, N);
	filter->data[4] = -8;
	Mat result = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/point.png", filter->len, filter->postion, filter->data, 2);
	result = abs(result);
	result.convertTo(result, CV_8U);
	threshold(result, result, 140, 255, 0);
	image_show(result, 2, "������ͼ");
}

//ֱ�߼��
void line_test() {
	//ˮƽ
	int M = 3;
	int N = 3;
	filter_screem<int>* filter_sp = (filter_screem<int>*)malloc(sizeof(filter_screem<int>));
	filter_sp->init(M, N);
	int src[9] = { -1,-1,-1,2,2,2,-1,-1,-1 };
	cudaMemcpy(filter_sp->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
	Mat result = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/m486.png", filter_sp->len, filter_sp->postion, filter_sp->data, 0.4);
	result.convertTo(result, CV_8U);
	threshold(result, result, 20, 255, 0);
	image_show(result, 0.4, "ˮƽ����ȡ");

	//��ֱ
	filter_sp = (filter_screem<int>*)malloc(sizeof(filter_screem<int>));
	filter_sp->init(M, N);
	int src_cz[9] = { -1,2,-1,-1,2,-1,-1,2,-1 };
	cudaMemcpy(filter_sp->data, src_cz, sizeof(int)*N*M, cudaMemcpyDefault);
	result = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/m486.png", filter_sp->len, filter_sp->postion, filter_sp->data, 0.4);
	result.convertTo(result, CV_8U);
	threshold(result, result, 20, 255, 0);
	image_show(result, 0.4, "��ֱ����ȡ");

	//+45��
	filter_sp = (filter_screem<int>*)malloc(sizeof(filter_screem<int>));
	filter_sp->init(M, N);
	int src_z45[9] = { 2,-1,-1,-1,2,-1,-1,-1,2 };
	cudaMemcpy(filter_sp->data, src_z45, sizeof(int)*N*M, cudaMemcpyDefault);
	result = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/m486.png", filter_sp->len, filter_sp->postion, filter_sp->data, 0.4);
	result.convertTo(result, CV_8U);
	threshold(result, result, 10, 255, 0);
	image_show(result, 0.4, "45��+����ȡ");

	//-45��
	filter_sp = (filter_screem<int>*)malloc(sizeof(filter_screem<int>));
	filter_sp->init(M, N);
	int src_f45[9] = { -1,-1,2,-1,2,-1,2,-1,-1 };
	cudaMemcpy(filter_sp->data, src_f45, sizeof(int)*N*M, cudaMemcpyDefault);
	result = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/m486.png", filter_sp->len, filter_sp->postion, filter_sp->data, 0.4);
	result.convertTo(result, CV_8U);
	threshold(result, result, 10, 255, 0);
	image_show(result, 0.4, "45��-����ȡ");
}

//10.2.5������Ե���
//��10.6
enum spacefilter_mode {
	prewitt_x = 0,
	prewitt_y = 1,

    sobel_x = 2,
	sobel_y =3,

	sobel_45z = 4,
	sobel_45f = 5,

	Laplace8=6,
	Gauss25=7,
	LoG=8
};

template<class T>
filter_screem<T>* set_filter(spacefilter_mode mode){
  filter_screem<T>* filter = (filter_screem<T>*)malloc(sizeof(filter_screem<T>));
  if (mode == 0)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { -1,0,1,-1,0,1,-1,0,1 };
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }

  if (mode == 1)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { -1,-1,-1,0,0,0,1,1,1 };
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }

  if (mode == 2)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { -1,0,1,-2,0,2,-1,0,1 };
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }


  if (mode == 3)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { -1,-2,-1,0,0,0,1,2,1};
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }

  if (mode == 4)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { 0,1,2,-1,0,1,-2,-1,0 };
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }

  if (mode == 5)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  int src[9] = { -2,-1,0,-1,0,1,0,1,2 };
	  cudaMemcpy(filter->data, src, sizeof(int)*N*M, cudaMemcpyDefault);
  }

  if (mode == 6)
  {
	  int M = 3;
	  int N = 3;
	  filter->init(M, N);
	  float src[9] = { 1.0,1.0,1.0,1.0,-8.0,1.0,1.0,1.0,1.0 };
	  cudaMemcpy(filter->data, src, sizeof(float)*N*M, cudaMemcpyDefault);
  }


  if (mode == 7)
  {
	  int var = 4;
	  int M = 6*var+1;
	  int N = 6*var+1;
	  int Mcenter = M / 2;
	  int Ncenter = N / 2;
	  float* src =(float* )malloc(sizeof(float)*M*N);
	  float sum=0;
	  for (size_t j = 0; j <M; j++)
		  { for (size_t i = 0; i <N; i++)
			    { 
		          src[j*N+i]= (float)(exp(-1*(pow((int)i - Ncenter,2.0)+pow((int)j - Mcenter,2.0))/(2.0*pow(var,2.0))));
				  sum = src[j*N + i] + sum;
				  //cout<< src[j*N + i] <<endl;
			    }
		  }

	  for (size_t j = 0; j < M; j++)
	  {
		  for (size_t i = 0; i < N; i++)
		  {
			  src[j*N + i] = src[j*N + i]/sum;
		
		  }
	  }
	  filter->init(M, N);
	  cudaMemcpy(filter->data, src, sizeof(float)*N*M, cudaMemcpyDefault);
  }


  if (mode == 8)
  {
	  int var = 4;
	  int M = 6 * var + 1;
	  int N = 6 * var + 1;
	  int Mcenter = M / 2;
	  int Ncenter = N / 2;
	  float* src = (float*)malloc(sizeof(float)*M*N);
	  float sum = 0;
	  for (size_t j = 0; j < M; j++)
	  {
		  for (size_t i = 0; i < N; i++)
		  {
			  src[j*N + i] = (float)((pow((int)i - Ncenter,2.0)+pow((int)j - Mcenter, 2.0)-2.0*var*var)
				  /pow(var,4.0)*exp(-1 * (pow((int)i - Ncenter, 2.0) + pow((int)j - Mcenter, 2.0)) / (2.0*pow(var, 2.0))));
			  sum = src[j*N + i] + sum;
			  //cout<< src[j*N + i] <<endl;
		  }
	  }

	  for (size_t j = 0; j < M; j++)
	  {
		  for (size_t i = 0; i < N; i++)
		  {
			  src[j*N + i] = src[j*N + i] / sum;

		  }
	  }
	  filter->init(M, N);
	  cudaMemcpy(filter->data, src, sizeof(float)*N*M, cudaMemcpyDefault);
  }

  return filter;
}

void two_fd_jd_test()
{   
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/house.png");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	image_show(Lena, 1, "ԭͼ");
	
	filter_screem<int>* filter_x = set_filter<int>(sobel_x);
	Mat result_x = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/house.png", filter_x->len, filter_x->postion, filter_x->data, 1);
	//cout << result_x << endl;
	
	filter_screem<int>* filter_y = set_filter<int>(sobel_y);
	Mat result_y = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/house.png", filter_y->len, filter_y->postion, filter_y->data, 1);
	
	filter_screem<int>* filter_45z = set_filter<int>(sobel_45z);
	Mat result_45z = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/house.png", filter_45z->len, filter_45z->postion, filter_45z->data, 1);

	filter_screem<int>* filter_45f = set_filter<int>(sobel_45f);
	Mat result_45f = space_filter_cpu<int>("C:/Users/Administrator/Desktop/opencv/house.png", filter_45f->len, filter_45f->postion, filter_45f->data, 1);

	Mat xy = result_x + result_y;
	xy.convertTo(xy, CV_8U);
	image_show(xy, 1, "sobel_y+soble_y");

	Mat xy_abs = abs(result_x) + abs(result_y);
	xy_abs.convertTo(xy_abs, CV_8U);
	//imshow("abs:sobel_y+soble_y",xy_abs);
	image_show(xy_abs, 1, "abs:sobel_y+soble_y");
	
	result_x=abs(result_x);
	//double max, min;
	//cv::Point min_loc, max_loc;
	//cv::minMaxLoc(result_x, &min, &max, &min_loc, &max_loc);
	//cout << max << endl;
	//cout << min << endl;
	result_x.convertTo(result_x, CV_8U);
	//imshow("sobel_x", result_x);
	image_show(result_x, 1, "sobel_x");

	result_y = abs(result_y);
	result_y.convertTo(result_y, CV_8U);
	//imshow("sobel_y", result_y);
	image_show(result_y, 1, "sobel_y");

	result_45z = abs(result_45z);
	result_45z.convertTo(result_45z, CV_8U);
	image_show(result_45z, 1, "sobel_45z");

	result_45f = abs(result_45f);
	result_45f.convertTo(result_45f, CV_8U);
	image_show(result_45f, 1, "sobel_45f");
}

//�㽻��
template<class arraytype>
texture <arraytype, cudaTextureType2D, cudaReadModeElementType> refTex_zero;//���ڼ���˫���Բ�ֵ

cudaArray* cuArray_zero;//����CUDA����

//ͨ����
template<class arraytype>
cudaChannelFormatDesc cuDesc_zero = cudaCreateChannelDesc<arraytype>();//ͨ����

////�ռ��˲�
template<class arraytype>
__global__ void zero_Kerkel(arraytype* pDstImgData, int imgHeight_des_d, int imgWidth_des_d,arraytype mark)
{   
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if(tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	  {     int idx = tidy * imgWidth_des_d + tidx;
		    pDstImgData[tidy * imgWidth_des_d + tidx] = 0;

			//if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
			//{
			//	int idx = tidy * imgWidth_des_d + tidx;
			//	pDstImgData[tidy * imgWidth_des_d + tidx] = 0;

			//	//	//����
			//	//	if ((int)(tex2D(refTex_zero, tidx - 1, tidy)*tex2D(refTex_zero, tidx + 1, tidy)) < 0 && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
			//	//	{
			//	//		if (tex2D(refTex_zero, tidx - 1, tidy) > 0 && tex2D(refTex_zero, tidx + 1, tidy) < 0)
			//	//		{
			//	//			if (abs(tex2D(refTex_zero, tidx + 1, tidy)) > mark)
			//	//			{
			//	//				pDstImgData[tidy * imgWidth_des_d + tidx + 1] = 255;
			//	//			}
			//	//			//if(abs(tex2D(refTex_zero, tidx-1, tidy))>mark)
			//	//			pDstImgData[tidy * imgWidth_des_d + tidx - 1] = 0;

			//	//		}
			//	//		else {

			//	//			//if (abs(tex2D(refTex_zero, tidx + 1, tidy)) > mark)
			//	//			pDstImgData[tidy * imgWidth_des_d + tidx + 1] = 0;
			//	//			if (abs(tex2D(refTex_zero, tidx - 1, tidy)) > mark)
			//	//				pDstImgData[tidy * imgWidth_des_d + tidx - 1] = 255;
			//	//			//printf("%d,%d,%d,%d \n", pDstImgData[tidy * imgWidth_des_d + tidx - 1], pDstImgData[tidy * imgWidth_des_d + tidx + 1], tex2D(refTex_zero, tidx - 1, tidy), tex2D(refTex_zero, tidx + 1, tidy));
			//	//		}

			//	//	}

			//	////����
			//	//if ((int)(tex2D(refTex_zero, tidx, tidy - 1)*tex2D(refTex_zero, tidx, tidy + 1)) < 0 && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
			//	//{
			//	//	if (tex2D(refTex_zero, tidx, tidy - 1) > 0 && tex2D(refTex_zero, tidx, tidy + 1) < 0)
			//	//	{
			//	//		//if (abs(tex2D(refTex_zero, tidx, tidy-1)) > mark)
			//	//		pDstImgData[(tidy - 1) * imgWidth_des_d + tidx] = 0;
			//	//		if (abs(tex2D(refTex_zero, tidx, tidy + 1)) > mark)
			//	//			pDstImgData[(tidy + 1) * imgWidth_des_d + tidx] = 255;
			//	//		//printf("%d,%d,%d,%d \n", pDstImgData[(tidy-1) * imgWidth_des_d + tidx ], pDstImgData[(tidy+1) * imgWidth_des_d + tidx], tex2D(refTex_zero, tidx, tidy-1), tex2D(refTex_zero, tidx, tidy+1));
			//	//	}
			//	//	else {

			//	//		if (abs(tex2D(refTex_zero, tidx, tidy - 1)) > mark)
			//	//		{
			//	//			pDstImgData[(tidy - 1) * imgWidth_des_d + tidx] = 255;
			//	//		}
			//	//		//if (abs(tex2D(refTex_zero, tidx, tidy + 1)) > mark)
			//	//		pDstImgData[(tidy + 1) * imgWidth_des_d + tidx] = 0;
			//	//	}

			////	}

			////	////45+
			////	if ((int)(tex2D(refTex_zero, tidx - 1, tidy + 1)*tex2D(refTex_zero, tidx + 1, tidy - 1)) < 0 && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
			////	{
			////		if (tex2D(refTex_zero, tidx + 1, tidy - 1) > 0 && tex2D(refTex_zero, tidx - 1, tidy + 1) < 0)
			////		{
			////			if (abs(tex2D(refTex_zero, tidx - 1, tidy + 1)) > mark)
			////			{
			////				pDstImgData[(tidy + 1) * imgWidth_des_d + tidx - 1] = 255;
			////			}
			////			//if (abs(tex2D(refTex_zero, tidx + 1, tidy - 1)) > mark)
			////			pDstImgData[(tidy - 1) * imgWidth_des_d + tidx + 1] = 0;
			////		}
			////		else {
			////			//if (abs(tex2D(refTex_zero, tidx - 1, tidy + 1)) > mark)
			////			pDstImgData[(tidy + 1) * imgWidth_des_d + tidx - 1] = 0;
			////			if (abs(tex2D(refTex_zero, tidx + 1, tidy - 1)) > mark)
			////				pDstImgData[(tidy - 1) * imgWidth_des_d + tidx + 1] = 255;
			////		}

			////	}

			////	////45-
			////	if ((int)(tex2D(refTex_zero, tidx - 1, tidy - 1)*tex2D(refTex_zero, tidx + 1, tidy + 1)) < 0 && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
			////	{
			////		if (tex2D(refTex_zero, tidx - 1, tidy - 1) > 0 && tex2D(refTex_zero, tidx + 1, tidy + 1) < 0)
			////		{
			////			//if (abs(tex2D(refTex_zero, tidx - 1, tidy - 1)) > mark)
			////			pDstImgData[(tidy - 1) * imgWidth_des_d + tidx - 1] = 0;
			////			if (abs(tex2D(refTex_zero, tidx + 1, tidy + 1)) > mark)
			////				pDstImgData[(tidy + 1) * imgWidth_des_d + tidx + 1] = 255;
			////		}
			////		else {
			////			if (abs(tex2D(refTex_zero, tidx - 1, tidy - 1)) > mark)
			////			{
			////				pDstImgData[(tidy - 1) * imgWidth_des_d + tidx - 1] = 255;
			////			}
			////			//if (abs(tex2D(refTex_zero, tidx + 1, tidy + 1)) > mark)
			////			pDstImgData[(tidy + 1) * imgWidth_des_d + tidx + 1] = 0;
			////		}
			////	}
			//}

				//����
				if((tex2D(refTex_zero<arraytype>, tidx-1, tidy)*tex2D(refTex_zero<arraytype>, tidx+1, tidy))<0 && abs(tex2D(refTex_zero<arraytype>, tidx - 1, tidy))- abs(tex2D(refTex_zero<arraytype>, tidx + 1, tidy))>mark && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
		          {    
							pDstImgData[idx] = 255;
			
				  }

				//����
				if((tex2D(refTex_zero<arraytype>, tidx, tidy-1)*tex2D(refTex_zero<arraytype>, tidx , tidy+1)) < 0 && abs(tex2D(refTex_zero<arraytype>, tidx, tidy-1)) - abs(tex2D(refTex_zero<arraytype>, tidx, tidy+1)) > mark && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
				  {
				
					pDstImgData[idx] = 255;
				  }

				////45+
				if((tex2D(refTex_zero<arraytype>, tidx-1, tidy+1)*tex2D(refTex_zero<arraytype>, tidx+1, tidy-1)) < 0 && abs(tex2D(refTex_zero<arraytype>, tidx - 1, tidy+1)) - abs(tex2D(refTex_zero<arraytype>, tidx + 1, tidy - 1)) > mark && 0<tidx && tidx<imgWidth_des_d-1 && 0 < tidy && tidy < imgHeight_des_d - 1)
				  {
					pDstImgData[idx] = 255;

				  }

				////45-
				if((tex2D(refTex_zero<arraytype>, tidx-1, tidy-1)*tex2D(refTex_zero<arraytype>, tidx+1, tidy+1)) < 0 && abs(tex2D(refTex_zero<arraytype>, tidx - 1, tidy - 1)) - abs(tex2D(refTex_zero<arraytype>, tidx + 1, tidy + 1)) > mark && 0 < tidx && tidx < imgWidth_des_d - 1 && 0 < tidy && tidy < imgHeight_des_d - 1)
				  {
					pDstImgData[idx] = 255;
				  }
		}

}

template<class arraytype>
Mat zero_crossing(Mat image,arraytype mark)
{
	Mat Lena = image.clone();
	if (typeid(arraytype) == typeid(int))
	{
		Lena.convertTo(Lena, CV_32S);
	}

	if (typeid(arraytype) == typeid(float))
	{
		Lena.convertTo(Lena, CV_32F);
	}

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_zero<arraytype>.addressMode[0] = cudaAddressModeClamp;
	refTex_zero<arraytype>.addressMode[1] = cudaAddressModeClamp;
	refTex_zero<arraytype>.normalized = false;
	refTex_zero<arraytype>.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_zero, &cuDesc_zero<arraytype>, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_zero<arraytype>, cuArray_zero);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_zero, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(arraytype), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	//Lena.convertTo(Lena, CV_32S);

	Mat dstImg1;
	if (typeid(arraytype) == typeid(int))
	{
		dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С
	}

	if (typeid(arraytype) == typeid(float))
	{
		dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32FC1);//��С
	}

	 
	//��������Ժ���cuda�ϵ�ͼ��
	arraytype* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(arraytype));
	t = cudaMemcpy(pDstImgData1, Lena.data, imgWidth_des_less * imgHeight_des_less * sizeof(arraytype)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	zero_Kerkel<arraytype><<<grid, block >>> (pDstImgData1, imgHeight_des_less, imgWidth_des_less,mark);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(arraytype)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_space_filter);
	cudaFree(pDstImgData1);
	return dstImg1.clone();
}
template Mat zero_crossing<int>(Mat,int);
template Mat zero_crossing<float>(Mat,float);

//ֱ������ LoG�˽��м��� ,��Դ����
void marrEdge(const Mat src, Mat& result, int kerValue,
	double delta)
                        {
	// ����LOG����
	Mat kernel;
	// �뾶
	int kerLen = kerValue / 2;
	kernel = Mat_<double>(kerValue, kerValue);
	// ����
	for (int i = -kerLen; i <= kerLen; i++)
	{
		for (int j = -kerLen; j <= kerLen; j++)
		{
			// ����������
			kernel.at<double>(i + kerLen, j + kerLen) =
				exp(-((pow(j, 2) + pow(i, 2)) /
				(pow(delta, 2) * 2)))
				* (((pow(j, 2) + pow(i, 2) - 2 *
					pow(delta, 2)) / (2 * pow(delta, 4))));
		}
	}

	// �����������
	int kerOffset = kerValue / 2;
	Mat laplacian = (Mat_<double>(src.rows - kerOffset * 2,
		src.cols - kerOffset * 2));
	result = Mat::zeros(src.rows - kerOffset * 2,
		src.cols - kerOffset * 2, src.type());
	double sumLaplacian;
	// ����������ͼ���Lapace����
	for (int i = kerOffset; i < src.rows - kerOffset; ++i)
	{
		for (int j = kerOffset; j < src.cols - kerOffset; ++j)
		{
			sumLaplacian = 0;
			for (int k = -kerOffset; k <= kerOffset; ++k)
			{
				for (int m = -kerOffset; m <= kerOffset; ++m)
				{
					// ����ͼ����
					sumLaplacian += src.at<uchar>(i + k, j + m) *
						kernel.at<double>(kerOffset + k,
							kerOffset + m);
				}
			}
			// ����Lapace���
			laplacian.at<double>(i - kerOffset,
				j - kerOffset) = sumLaplacian;
		}
	}

	//// ����㽻�� Ѱ�ұ�Ե����
	for (int y = 1; y < result.rows - 1; ++y)
	{
		for (int x = 1; x < result.cols - 1; ++x)
		{
			result.at<uchar>(y, x) = 0;
			// �����ж�
			if (laplacian.at<double>(y - 1, x) *
				laplacian.at<double>(y + 1, x) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y, x - 1) *
				laplacian.at<double>(y, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y + 1, x - 1) *
				laplacian.at<double>(y - 1, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y - 1, x - 1) *
				laplacian.at<double>(y + 1, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
		}
	}
}


//-------------------------------�����㷨��Ե����㷨-----------------------------------
//���� �ݶȷ����
Mat normal_detect(const Mat& image_N) {
	Mat image = image_N.clone();
	image.convertTo(image_N, CV_32F);
	int sp = 1;
	int f45 = 2;
	int cz = 3;
	int z45 = 4;
	
	int M = image.rows;
	int N = image.cols;
	
	for (size_t i = 0; i <M; i++)
		{  for (size_t j = 0; j <N; j++)
			  { 
		        float value=image.at<float>(i, j);
				if (337.5< value  || value <= 22.5 || (value <=202.5&& 157.5 < value))
					image.at<float>(i, j) = 1;
				
                if ((202.5 < value && value <= 247.5) || (value <= 67.5  && 22.5< value))
					image.at<float>(i, j) = 2;
				
				if ((67.5 < value  && value <= 112.5) || (value <=292.5 && 247.5 < value))
					image.at<float>(i, j) = 3;
				
				if ((112.5 < value  && value <= 157.5) || (value <=337.5 && 292.5 < value))
					image.at<float>(i, j) = 4;
			  }
		}
	return image.clone();
}

//���Ʒ�����Ե��
Mat limit_big(const Mat& image_N,const Mat& direct) {
	Mat image = image_N.clone();
	image.convertTo(image_N, CV_32F);
    int M = image.rows;
    int N = image.cols;
	Mat gN = Mat::zeros(image.size(),CV_32F);

	for (size_t i = 1; i <M-1; i++)
		{  for (size_t j = 1; j <N-1; j++)
			   { 
				 if(direct.at<float>(i, j) == 1.0)//1
					{
					   float vlaue = image.at<float>(i, j);
					   if (vlaue < image.at<float>(i + 1, j) || vlaue < image.at<float>(i - 1, j))
						   gN.at<float>(i, j) = 0;
					   else
						   gN.at<float>(i, j) = vlaue;
					}

				 if (direct.at<float>(i, j) == 2.0)//1
				 {
					 float vlaue = image.at<float>(i, j);
					 if (vlaue < image.at<float>(i -1, j-1) || vlaue < image.at<float>(i + 1, j+1))
						 gN.at<float>(i, j) = 0;
					 else
						 gN.at<float>(i, j) = vlaue;
				 }

				 if (direct.at<float>(i, j) == 3.0)//1
				 {
					 float vlaue = image.at<float>(i, j);
					 if (vlaue < image.at<float>(i, j - 1) || vlaue < image.at<float>(i, j + 1))
						 gN.at<float>(i, j) = 0;
					 else
						 gN.at<float>(i, j) = vlaue;
				 }

				 if (direct.at<float>(i, j) == 4.0)//1
				 {
					 float vlaue = image.at<float>(i, j);
					 if (vlaue < image.at<float>(i-1, j + 1) || vlaue < image.at<float>(i+1, j - 1))
						 gN.at<float>(i, j) = 0;
					 else
						 gN.at<float>(i, j) = vlaue;
				 }

			   }
		}
	return gN.clone();
}


Mat threashold_rato(Mat& e_N,float rato) 
{
	Mat e = e_N.clone();
	double max, min;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(e, &min, &max, &min_loc, &max_loc);
	/*cout << "max:" << max << endl;
	cout << "min:" << min << endl;*/
	int max_N = (int)max * rato;
	cout << "max_N:" << max_N<< endl;
	threshold(e, e, max_N, 255, 0);
	return e.clone();
}

//��Ե�ӳ�������8��ͨ
Mat Edge_lengthening(Mat& gNH, Mat& gNL,Mat& mark) {
	Mat result=Mat::zeros(gNH.size(),CV_8U);
	int M = gNH.rows;
	int N = gNH.cols;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (gNH.at<uchar>(i, j) == 255)
			{ //8��ͨ����
				if (gNL.at<uchar>(i - 1, j - 1) > 0 && mark.at<uchar>(i - 1, j - 1) == 0)//gNL������δ�궨����==255
				{
					result.at<uchar>(i - 1, j - 1) = 255;
					mark.at<uchar>(i - 1, j - 1) = 1;
				}

				if (gNL.at<uchar>(i - 1, j) > 0 && mark.at<uchar>(i - 1, j) == 0)
				{
					result.at<uchar>(i - 1, j) = 255;
					mark.at<uchar>(i - 1, j) = 1;
				}

				if (gNL.at<uchar>(i - 1, j + 1) > 0 && mark.at<uchar>(i - 1, j + 1) == 0)
				{
					result.at<uchar>(i - 1, j + 1) = 255;
					mark.at<uchar>(i - 1, j + 1) = 1;
				}

				if (gNL.at<uchar>(i, j - 1) > 0 && mark.at<uchar>(i, j - 1) == 0)
				{
					result.at<uchar>(i, j - 1) = 255;
					mark.at<uchar>(i, j - 1) = 1;
				}


				if (gNL.at<uchar>(i, j + 1) > 0 && mark.at<uchar>(i, j + 1) == 0)
				{
					result.at<uchar>(i, j + 1) = 255;
					mark.at<uchar>(i, j + 1) = 1;
				}

				if (gNL.at<uchar>(i + 1, j - 1) > 0 && mark.at<uchar>(i + 1, j - 1) == 0)
				{
					result.at<uchar>(i + 1, j - 1) = 255;
					mark.at<uchar>(i + 1, j - 1) = 1;
				}

				if (gNL.at<uchar>(i + 1, j) > 0 && mark.at<uchar>(i + 1, j) == 0)
				{
					result.at<uchar>(i + 1, j) = 255;
					mark.at<uchar>(i + 1, j) = 1;
				}

				if (gNL.at<uchar>(i + 1, j + 1) > 0 && mark.at<uchar>(i + 1, j + 1) == 0)
				{
					result.at<uchar>(i + 1, j + 1) = 255;
					mark.at<uchar>(i + 1, j + 1) = 1;
				}

				//if (gNL.at<uchar>(i - 1, j - 1) > 0)//gNL������δ�궨����==255
				//{
				//	result.at<uchar>(i - 1, j - 1) = 255;
				//	gNL.at<uchar>(i - 1, j - 1) = 0;
				//}

				//if (gNL.at<uchar>(i - 1, j) > 0)
				//{
				//	result.at<uchar>(i - 1, j) = 255;
				//	gNL.at<uchar>(i - 1, j) = 0;
				//}

				//if (gNL.at<uchar>(i - 1, j + 1) > 0)
				//{
				//	result.at<uchar>(i - 1, j + 1) = 255;
				//	gNL.at<uchar>(i - 1, j + 1) == 0;
				//}

				//if (gNL.at<uchar>(i, j - 1) > 0 )
				//{
				//	result.at<uchar>(i, j - 1) = 255;
				//	gNL.at<uchar>(i, j - 1) == 0;
				//}


				//if (gNL.at<uchar>(i, j + 1) > 0 )
				//{
				//	result.at<uchar>(i, j + 1) = 255;
				//	gNL.at<uchar>(i, j + 1) == 0;
				//}

				//if (gNL.at<uchar>(i + 1, j - 1) > 0)
				//{
				//	result.at<uchar>(i + 1, j - 1) = 255;
				//	gNL.at<uchar>(i + 1, j - 1) == 0;
				//}

				//if (gNL.at<uchar>(i + 1, j) > 0)
				//{
				//	result.at<uchar>(i + 1, j) = 255;
				//	gNL.at<uchar>(i + 1, j) == 0;
				//}

				//if (gNL.at<uchar>(i + 1, j + 1) > 0)
				//{
				//	result.at<uchar>(i + 1, j + 1) = 255;
				//	gNL.at<uchar>(i + 1, j + 1) == 0;
				//}
			}
		}
	}
	return result.clone();
}

//��������
//void canny_test() {
//	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/house2.png");
//	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
//	image_show(Lena, 0.4, "ԭͼ");
//
//	//һ���ȸ�ͨ�˲�����������˹�任
//	filter_screem<float>* filter_G = set_filter<float>(Gauss25);
//	Mat GSmat = space_filter_cpu_mat(Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
//	image_show(GSmat, 0.4, "��˹�˲���");
//
//	filter_screem<int>* filter_x = set_filter<int>(sobel_x);
//	Mat soble_x = space_filter_cpu_mat(GSmat, filter_x->len, filter_x->postion, filter_x->data, 1);
//	image_show(soble_x, 0.4, "soble_x");
//
//	filter_screem<int>* filter_y = set_filter<int>(sobel_y);
//	Mat soble_y = space_filter_cpu_mat(GSmat, filter_y->len, filter_y->postion, filter_y->data, 1);
//	image_show(soble_y, 0.4, "soble_y");
//	
//	//�����ݶȷ���
//	soble_x.convertTo(soble_x, CV_32F);
//	soble_y.convertTo(soble_y, CV_32F);
//	//cout << soble_y << endl;
//	Mat M_xy;
//	sqrt(soble_x.mul(soble_x) + soble_y.mul(soble_y), M_xy);
//	//cout << M_xy << endl;
//	
//	//���㷽��ת��Ϊ�Ƕ�ֵ
//	Mat arctan;
//	phase(soble_y, soble_x,arctan);
//	arctan = arctan * 180 / 3.1415926;
//	
//	//Ѱ����ӽ��Ƕȵķ���dk
//	Mat direct=normal_detect(arctan);
//	
//	//һֱ���ݶ�ͼ��Ӧ�÷��������
//	Mat mide = limit_big(M_xy, direct);
//	image_show(mide, 0.4, "�м���");
//	//cout<<mide<<endl;
//
//	//����gNL��gNH
//	Mat mide_H = threashold_rato(mide, 0.10);//gNH
//	image_show(mide_H, 0.4, "gNH");
//	Mat mide_L = threashold_rato(mide, 0.04);//gNL
//	image_show(mide_L, 0.4, "gNL");
//	mide_L= mide_L - mide_H;//gNL=gNL-gNH
//	image_show(mide_L, 0.4, "gNL-gNH");
//
//	//�����ӳ�
//	mide_H.convertTo(mide_H, CV_8U);
//	mide_L.convertTo(mide_L, CV_8U);
//	Mat mark = Mat::zeros(mide_H.size(), CV_8U);
//	Mat gNH_increas_N=Edge_lengthening(mide_H,mide_L,mark);
//	Mat gNH_increas= Mat::zeros(mide_H.size(), CV_8U);
//	add(mide_H, gNH_increas_N, mide_H);
//	
//	int i = 0;
//	Scalar ss = sum(gNH_increas_N);
//	while(ss[0]>0){
//	      gNH_increas = Edge_lengthening(gNH_increas_N,mide_L,mark);
//		  add(mide_H,gNH_increas,mide_H);
//		  gNH_increas_N = gNH_increas.clone();
//		  ss = sum(gNH_increas);
//		  i = i + 1;
//	}
//
//	image_show(mide_H, 0.4, "fg_output");
//
//	//opencv �Դ���canny
//
//	Mat output;
//	//cv::Canny(Lena,output,7, 17,3,true);
//	soble_x.convertTo(soble_x,CV_16SC1);
//	soble_y.convertTo(soble_y, CV_16SC1);
//	cv::Canny(soble_x, soble_y, output, 7, 17,true);
//	image_show(output, 0.4, "opencv_output");
//}


void Edge_Connection_test() {
    


}

void chapter10_test()
{   //single_point();
	//line_test();
	//two_fd_jd_test();
	//set_Marr_Hildreth_filter();
    // LoG_test();
    //canny_test();
};


