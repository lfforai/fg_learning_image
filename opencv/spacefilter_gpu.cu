#include "spacefilter_gpu.cuh"

template<class datatype>
f_screem<datatype>* set_f(sf_mode mode) {
	f_screem<datatype>* filter = (f_screem<datatype>*)malloc(sizeof(f_screem<datatype>));
	if (mode == 0)
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { -1,0,1,-1,0,1,-1,0,1 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 1)
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { -1,-1,-1,0,0,0,1,1,1 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 2)//x
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { -1,0,1,-2,0,2,-1,0,1 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}


	if (mode == 3)//y
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { -1,-2,-1,0,0,0,1,2,1 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 4)
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { 0,1,2,-1,0,1,-2,-1,0 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 5)
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { -2,-1,0,-1,0,1,0,1,2 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 6)
	{
		int M = 3;
		int N = 3;
		filter->init(M, N);
		datatype src[9] = { 1.0,1.0,1.0,1.0,-8.0,1.0,1.0,1.0,1.0 };
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	
	if (mode == 7)
	{
		int var = 4;
		int M = 6 * var + 1;
		int N = 6 * var + 1;
		int Mcenter = M / 2;
		int Ncenter = N / 2;
		datatype* src = (datatype*)malloc(sizeof(datatype)*M*N);
		datatype sum = 0;
		for (size_t j = 0; j < M; j++)
		{
			for (size_t i = 0; i < N; i++)
			{
				src[j*N + i] = (datatype)(exp(-1 * (pow((int)i - Ncenter, 2.0) + pow((int)j - Mcenter, 2.0)) / (2.0*pow(var, 2.0))));
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
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}


	if (mode == 8)
	{
		int var = 4;
		int M = 6 * var + 1;
		int N = 6 * var + 1;
		int Mcenter = M / 2;
		int Ncenter = N / 2;
		datatype* src = (datatype*)malloc(sizeof(datatype)*M*N);
		datatype sum = 0;
		for (size_t j = 0; j < M; j++)
		{
			for (size_t i = 0; i < N; i++)
			{
				src[j*N + i] = (datatype)((pow((int)i - Ncenter, 2.0) + pow((int)j - Mcenter, 2.0) - 2.0*var*var)
					/ pow(var, 4.0)*exp(-1 * (pow((int)i - Ncenter, 2.0) + pow((int)j - Mcenter, 2.0)) / (2.0*pow(var, 2.0))));
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
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	if (mode == 9)//均值模板
	{
		int M = 5;
		int N = 5;
		filter->init(M, N);
		int len = M * N;
		datatype* src =(datatype *)malloc(M*N*sizeof(datatype));
		for (size_t i = 0; i <len ; i++)
		{src[i]=1.0/25.0;}
		cudaMemcpy(filter->data, src, sizeof(datatype)*N*M, cudaMemcpyDefault);
	}

	return filter;
}

template<class texturetpye>
texture <texturetpye, cudaTextureType2D, cudaReadModeElementType> refTex_space_f;//用于计算双线性插值

cudaArray* cuArray_space_f;//声明CUDA数组

//通道数
template<class texturetpye>
cudaChannelFormatDesc cuDesc_space_f = cudaCreateChannelDesc<texturetpye>();//通道数

template<class datatype,class arraytype>
__device__ arraytype space_f(int x, int y, Point_f* point_gpu, datatype* data, int len) {
	int x_N;
	int y_N;
	arraytype result = 0;
	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		result = result + (arraytype)((tex2D(refTex_space_f<arraytype>, x_N, y_N))*((arraytype)data[i]));
		//if (x == 0 && y == 0)
				//printf("x:%d,y:%d,|%d,%f,%d,%f \n",x_N,y_N,i, (((float)tex2D(refTex_space_filter, x_N, y_N))*((float)data[i])),(int)tex2D(refTex_space_filter, x_N, y_N),data[i]);
	}
	return  result;
}

////空间滤波
template<class datatype,class arraytype>
__global__ void space_f_Kerkel(arraytype* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_f* point_gpu, datatype* data, int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		pDstImgData[idx] = space_f<datatype,arraytype>(tidx, tidy, point_gpu, data, len);
	}
}

template<class datatype,class arraytype>
Mat space_filter_gpu(char * path,Mat& image, int len, Point_f*  point_offset_N, datatype* data, float size)
{   
	Mat Lena;
	if (strlen(path)==0)
	{
		Lena = image.clone();
	}
	else{
		Lena = imread(path);
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
		image_show(Lena, size, "原图");
	}

	if (typeid(arraytype) == typeid(int)) {
		Lena.convertTo(Lena,CV_32S);
	}

	if (typeid(arraytype) == typeid(float)) {
		Lena.convertTo(Lena, CV_32F);
	}

	int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//原图像宽
	int imgHeight_src = Lena.rows;//原图像高
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

	//设置1纹理属性
	cudaError_t t;
	refTex_space_f<arraytype>.addressMode[0] = cudaAddressModeClamp;
	refTex_space_f<arraytype>.addressMode[1] = cudaAddressModeClamp;
	refTex_space_f<arraytype>.normalized = false;
	refTex_space_f<arraytype>.filterMode = cudaFilterModePoint;
	//绑定cuArray到纹理
	cudaMallocArray(&cuArray_space_f, &cuDesc_space_f<arraytype>, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_space_f<arraytype>, cuArray_space_f);
	//拷贝数据到cudaArray
	t = cudaMemcpyToArray(cuArray_space_f, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(arraytype), cudaMemcpyHostToDevice);

	//输出放缩以后在cpu上图像
	Mat dstImg1; 
	if (typeid(arraytype) == typeid(int)) {
		dstImg1=Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//缩小
	}

	if (typeid(arraytype) == typeid(float)) {
		dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32FC1);//缩小
	}

	//输出放缩以后在cuda上的图像
	arraytype* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(arraytype));
	t = cudaMemcpy(pDstImgData1, Lena.data, imgWidth_des_less * imgHeight_des_less * sizeof(arraytype)*channels, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	space_f_Kerkel<datatype,arraytype> << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, point_offset_N, data, len);
	cudaDeviceSynchronize();

	//从GPU拷贝输出数据到CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(arraytype)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_space_f);
	cudaFree(pDstImgData1);
	//namedWindow("cuda_point最近插值：", WINDOW_NORMAL);
	//imshow("腐蚀以后的图像：", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
	//Lena = Lena - dstImg1;
	//Lena.convertTo(Lena, CV_8U);
	//imshow("地球北极_拉普拉斯变换后的图", Lena);
	//result.convertTo(result, CV_32F);
	//image_show(Lena, 1, "变化后的图");
	return dstImg1.clone();
}
template Mat space_filter_gpu<float,float>(char*,Mat&,int, Point_f*, float* ,float);

//返回计算sobel算子梯度
Mat sobel_grad(Mat& image_N,int mode=0) {
	Mat image = image_N.clone();
	image.convertTo(image, CV_32F);
	f_screem<float>* filter_x = set_f<float>(sf_mode::sobel_x_N);
	Mat soble_x = space_filter_gpu<float, float>("", image, filter_x->len, filter_x->postion, filter_x->data, 1);
	//image_show(soble_x, 0.4, "soble_x");

	f_screem<float>* filter_y = set_f<float>(sf_mode::sobel_y_N);
	Mat soble_y = space_filter_gpu<float, float>("", image, filter_y->len, filter_y->postion, filter_y->data, 1);
	//image_show(soble_y, 0.4, "soble_y");

	//计算梯度幅度
	soble_x.convertTo(soble_x, CV_32F);
	soble_y.convertTo(soble_y, CV_32F);

	Mat M_xy;
	if(mode == 0)//
	sqrt(soble_x.mul(soble_x) + soble_y.mul(soble_y), M_xy);
	if(mode == 1)//
	M_xy=abs(soble_x) + abs(soble_y);
	return M_xy.clone();
}

float Max_ofmat(Mat& H) {
	double max, min;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);
	return max;
}

void LoG_test()
{
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/house.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	image_show(Lena, 0.5, "原图");
	//Lena.convertTo(Lena, CV_32F, 1.0/255.0);
	//cout << Lena << endl;

	//Mat result;
	//marrEdge(Lena, result, 25, 4);
	//image_show(result,1,"GSmat");

	//先高通滤波，后拉普拉斯变换
	f_screem<float>* filter_G = set_f<float>(sf_mode::Gauss25_N);
	Mat GSmat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	
	f_screem<float>* filter_x = set_f<float>(sf_mode::Laplace8_N);//Laplace8);
	Mat result_x = space_filter_gpu<float, float>("", GSmat, filter_x->len, filter_x->postion, filter_x->data, 1);
	//result_x.convertTo(result_x, CV_32S);
	
	//直接计算LOG算子
	//f_screem<float>* filter_x = set_f<float>(sf_mode::LoG_N);
	//Mat result_x = space_filter_gpu<float,float>("",Lena, filter_x->len, filter_x->postion, filter_x->data, 1);
	//result_x.convertTo(result_x, CV_32S,255);

	//Mat show = result_x.clone();
	//demarcate(show);
	//image_show(show, 0.5, "show");

	Mat e = result_x.clone();

	float mark = 0;
	result_x = zero_crossing<float>(result_x,mark);

	result_x.convertTo(result_x, CV_8U);
	//imshow("HARR-HILL-zeros", result_x);
	image_show(result_x, 0.5, "HARR-HILL-zeros");

	//double max, min;
	//cv::Point min_loc, max_loc;
	//cv::minMaxLoc(e, &min, &max, &min_loc, &max_loc);
	//cout << "max:" << max << endl;
	//cout << "min:" << min << endl;
	//int max_N = (int)max * 0.04;
	//threshold(e, e, max_N, 255, 0);
	mark = 1;
	e = zero_crossing<float>(e, mark);
	e.convertTo(e, CV_8U);
	image_show(e, 0.5, "HARR-HILL-threshold");
}

void canny_test() {
	
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/house.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	image_show(Lena, 0.4, "原图");

	//一、先高通滤波，后拉普拉斯变换
	f_screem<float>* filter_G = set_f<float>(sf_mode::Gauss25_N);
	Mat GSmat = space_filter_gpu<float,float>("",Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(GSmat, 0.4, "高斯滤波器");

	f_screem<float>* filter_x = set_f<float>(sf_mode::sobel_x_N);
	Mat soble_x = space_filter_gpu<float,float>("",GSmat, filter_x->len, filter_x->postion, filter_x->data, 1);
	image_show(soble_x, 0.4, "soble_x");

	f_screem<float>* filter_y = set_f<float>(sf_mode::sobel_y_N);
	Mat soble_y = space_filter_gpu<float,float>("",GSmat, filter_y->len, filter_y->postion, filter_y->data, 1);
	image_show(soble_y, 0.4, "soble_y");

	//计算梯度幅度
	soble_x.convertTo(soble_x, CV_32F);
	soble_y.convertTo(soble_y, CV_32F);
	
	long start, end;
	start = clock();
	//cout << soble_y << endl;
	Mat M_xy;
	sqrt(soble_x.mul(soble_x) + soble_y.mul(soble_y), M_xy);
	//cout << M_xy << endl;

	//计算方向，转换为角度值
	Mat arctan;
	phase(soble_y, soble_x, arctan);
	arctan = arctan * 180 / 3.1415926;

	//寻找最接近角度的方向dk
	Mat direct = normal_detect(arctan);

	//一直对梯度图像应用非最大限制
	Mat mide = limit_big(M_xy, direct);
	//image_show(mide, 0.4, "中间结果");

	//计算gNL，gNH
	Mat mide_H = threashold_rato(mide, 0.10);//gNH
	image_show(mide_H, 0.4, "gNH");
	Mat mide_L = threashold_rato(mide, 0.04);//gNL
	image_show(mide_L, 0.4, "gNL");
	mide_L = mide_L - mide_H;//gNL=gNL-gNH
	image_show(mide_L, 0.4, "gNL-gNH");

	//短线延长
	mide_H.convertTo(mide_H, CV_8U);
	mide_L.convertTo(mide_L, CV_8U);
	Mat mark = Mat::zeros(mide_H.size(), CV_8U);
	Mat gNH_increas_N = Edge_lengthening(mide_H, mide_L, mark);
	Mat gNH_increas = Mat::zeros(mide_H.size(), CV_8U);
	add(mide_H, gNH_increas_N, mide_H);

	int i = 0;
	Scalar ss = sum(gNH_increas_N);
	while (ss[0] > 0 && i<5) {
		gNH_increas = Edge_lengthening(gNH_increas_N, mide_L, mark);
		add(mide_H, gNH_increas, mide_H);
		gNH_increas_N = gNH_increas.clone();
		ss = sum(gNH_increas);
		i = i + 1;
	}
	//测试的程序段
	end = clock();
	cout<<"cuda耗时：" << start - end <<endl;

	image_show(mide_H, 0.5, "fg_canny_output");

	//二、opencv 自带的canny
	start = clock();
	Mat output;
	//cv::Canny(Lena,output,7, 17,3,true);
	soble_x.convertTo(soble_x, CV_16SC1);
	soble_y.convertTo(soble_y, CV_16SC1);
	cv::Canny(soble_x, soble_y, output, 6, 15, true);
	end = clock();
	cout << "opencv耗时：" <<start-end<< endl;
	image_show(output, 0.5, "opencv_canny_output");
}

//边缘连接和边界检测
//局部处理
void Partial_treatment() {
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/car.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	image_show(Lena,1, "原图");

	//一、先高通滤波，后拉普拉斯变换
	Mat GSmat=Lena.clone();
	//f_screem<float>* filter_G = set_f<float>(sf_mode::Gauss25_N);
	//Mat GSmat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(GSmat, 0.4, "高斯滤波器");

	f_screem<float>* filter_x = set_f<float>(sf_mode::sobel_x_N);
	Mat soble_x = space_filter_gpu<float, float>("", GSmat, filter_x->len, filter_x->postion, filter_x->data, 1);
	image_show(soble_x,1, "soble_x");

	f_screem<float>* filter_y = set_f<float>(sf_mode::sobel_y_N);
	Mat soble_y = space_filter_gpu<float, float>("", GSmat, filter_y->len, filter_y->postion, filter_y->data, 1);
	image_show(soble_y,1, "soble_y");

	//计算梯度幅度
	soble_x.convertTo(soble_x, CV_32F);
	soble_y.convertTo(soble_y, CV_32F);

	//梯度和角度
	Mat arctan;
	phase(soble_y, soble_x, arctan);
	arctan = arctan * 180.0/3.1415926;
	
	Mat M_value;
	sqrt(soble_x.mul(soble_x) + soble_y.mul(soble_y), M_value);
	image_show(M_value, 1, "M_value");
	//M_value.convertTo(M_value, CV_8U);
	
	//TM最大梯度的30%
	double max, min;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(M_value, &min, &max, &min_loc, &max_loc);
	cout<<max<<endl;
	float TM = (float)max * 0.3;
	cout<<TM<<endl;
	
	//行
	float A_90 = 180;
	float A_270 = 360;
	float TA = 45;

	Mat g_90 = Mat::zeros(Lena.size(),CV_8U);
	int M = Lena.rows;
	int N = Lena.cols;
		for (size_t i = 0; i <M; i++)
		  {  for (size_t j = 0; j <N; j++)
			   { 
			      if(M_value.at<float>(i, j) > TM && arctan.at<float>(i,j)>=A_90-TA && arctan.at<float>(i, j) <=A_90+TA)
					 g_90.at<uchar>(i, j)=255;
			   }
		  }
    image_show(g_90, 1, "M_180");
	
    Mat g_270 = Mat::zeros(Lena.size(), CV_8U);
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (M_value.at<float>(i, j) > TM && (arctan.at<float>(i, j) >= A_270 - TA || arctan.at<float>(i, j) <= TA))
				g_270.at<uchar>(i, j) = 255;
		}
	}
	image_show(g_270, 1, "M_360");

	Mat g=OR_two(g_270,g_90,0,255);
	image_show(g, 1, "MR");
	g.convertTo(g, CV_8U);

	Mat result=Mat::zeros(Lena.size(), CV_8U);
	int k = 25;
	for (size_t i = 0; i < M; i++)
		{
			for (size_t j = 0; j < N; j++)
			{   
				if (j < N - k)
				{
					if (g.at<uchar>(i, j) == 255)
					{
						result.at<uchar>(i, j) = 255;

						int k_n = 1;
						while (k_n < k)
						{
							if (g.at<uchar>(i, j + k_n) == 255)
								break;
							else
								k_n = k_n + 1;
						}

						if (k_n > 1 && g.at<uchar>(i, j + k_n) == 255 && k_n<k)
						{
							for(size_t n = 1; n <= k_n; n++)
							   {
								result.at<uchar>(i, j + (int)n) = 255;
							   }
						}
					}
				}
				else {
				    
					result.at<uchar>(i, j) = g.at<uchar>(i, j);
				}
			}
		}

	image_show(result, 1, "M_row");

	//列
	A_90 = 90;
	A_270 = 270;

	g_90 = Mat::zeros(Lena.size(), CV_8U);
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (M_value.at<float>(i, j) > TM && arctan.at<float>(i, j) >= A_90 - TA && arctan.at<float>(i, j) <= A_90 + TA)
				g_90.at<uchar>(i, j) = 255;
		}
	}
	image_show(g_90, 1, "ML_90");

	g_270 = Mat::zeros(Lena.size(), CV_8U);
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			if (M_value.at<float>(i, j) > TM && arctan.at<float>(i, j) >= A_270 - TA && arctan.at<float>(i, j) <= A_270 + TA)
				g_270.at<uchar>(i, j) = 255;
		}
	}
	image_show(g_270, 1, "ML_270");

	g = OR_two(g_270, g_90, 0, 255);
	image_show(g, 1, "ML");
	g.convertTo(g, CV_8U);

	Mat result_N = Mat::zeros(Lena.size(), CV_8U);

	k = 25;
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			if (j < M - k)
			{
				if (g.at<uchar>(j, i) == 255)
				{
					result_N.at<uchar>(j, i) = 255;
					int k_n = 1;
					while (k_n < k)
					{
						if (g.at<uchar>(j + k_n,i) == 255)
							break;
						else
							k_n = k_n + 1;
					}

					if (k_n > 1 && g.at<uchar>(j + k_n,i) == 255 && k_n<k)
					{
						for (size_t n = 1; n <= k_n; n++)
						{
							result_N.at<uchar>(j + (int)n,i) = 255;
						}
					}
				}
			}
			else {

				result_N.at<uchar>(j, i) = g.at<uchar>(j, i);
			}
		}
	}
	image_show(result_N, 1, "M_col");

	Mat line=OR_two(result_N, result, 0, 255); 
	image_show(line, 1, "line");
}