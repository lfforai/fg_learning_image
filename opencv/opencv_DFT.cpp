﻿#include "opencv_DFT.h"

//基于opencv的频谱、相位图画法
#define PI2 2*3.141592654

//一、通用自己编写的 通用opencv API
//对比两个mat是否相等，来源于网上
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
	if (mat1.empty() && mat2.empty()) {
		return true;
	}
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
		mat1.channels() != mat2.channels()) {
		return false;
	}
	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
		return false;
	}
	int nrOfElements1 = mat1.total()*mat1.elemSize();
	if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
	return lvRet;
}

//来自微博https ://blog.csdn.net/cyf15238622067/article/details/88231590 
//傅里叶逆变换
Mat fourior_inverser(Mat &src_img,cv::Mat &real_img, cv::Mat &ima_img)
{   
	if (src_img.channels() == 1)
	{
		src_img.convertTo(src_img, CV_32FC1);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
		Mat padded;
		copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));

		Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
		Mat complexI;
		merge(temp, 2, complexI);
		idft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//显示频谱图
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else if (src_img.channels() == 2)
	{
		src_img.convertTo(src_img, CV_32FC2);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);

		Mat temp[] = { Mat::zeros(src_img.size(),CV_32FC1),Mat::zeros(src_img.size(),CV_32FC1) };
		split(src_img, temp);//切分为实部和虚部

		Mat padded_real;//实部按dft要求扩展尺寸
		copyMakeBorder(temp[0], padded_real, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		Mat padded_imal;//虚部按dft要求扩展尺寸
		copyMakeBorder(temp[1], padded_imal, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		temp[0] = padded_real;
		temp[1] = padded_imal;

		Mat complexI;
		//扩展以后再合并
		merge(temp, 2, complexI);
		idft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//分解为实部和虚部
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else {
		cout << "报错,输入的Mat不能为3通道!" << endl;
		return Mat::zeros(2, 2, CV_32FC2);
	}
}

//输入的图像是一个单通道（双通道）的傅里叶变换 
Mat fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	if (src_img.channels() == 1)
	{   
		src_img.convertTo(src_img, CV_32FC1);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
		Mat padded;
		copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));

		Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
		Mat complexI;
		merge(temp, 2, complexI);
		dft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//显示频谱图
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else if(src_img.channels() == 2)
   {
		src_img.convertTo(src_img, CV_32FC2);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
	    
		Mat temp[] = {Mat::zeros(src_img.size(),CV_32FC1),Mat::zeros(src_img.size(),CV_32FC1)};
		split(src_img,temp);//切分为实部和虚部

		Mat padded_real;//实部按dft要求扩展尺寸
		copyMakeBorder(temp[0], padded_real, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		Mat padded_imal;//虚部按dft要求扩展尺寸
		copyMakeBorder(temp[1], padded_imal, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi
		
		temp[0] = padded_real;
		temp[1] = padded_imal;

		Mat complexI;
		//扩展以后再合并
		merge(temp, 2, complexI);
		dft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//分解为实部和虚部
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else {
		cout<<"报错,输入的Mat不能为3通道!"<<endl;
		return Mat::zeros(2, 2, CV_32FC2);
	}
	
}

//图像中心化才能护理
void move_to_center(Mat &center_img)
{
	int cx = center_img.cols / 2;
	int cy = center_img.rows / 2;
	Mat q0(center_img, Rect(0, 0, cx, cy));// Top-Left - Create a ROI per quadrant
	Mat q1(center_img, Rect(cx, 0, cx, cy));// Top-Right
	Mat q2(center_img, Rect(0, cy, cx, cy));// Bottom-Left
	Mat q3(center_img, Rect(cx, cy, cx, cy));// Bottom-Right

	Mat tmp;// swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);// swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

//返回log以后的频谱图[0,255]
void amplitude_common(Mat &center_img){
	center_img.convertTo(center_img,CV_32FC2);
	Mat temp[] = {Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	magnitude(temp[0], temp[1],center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回普通的频谱图[0.255]
void amplitude_log(Mat &center_img) {
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	magnitude(temp[0], temp[1], center_img);
	
	center_img += Scalar::all(1);
	log(center_img, center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回log以后的想位图[0,255]
void angle_common(Mat &center_img){
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	phase(temp[0], temp[1], center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回普通的相位图[0.255]
void angle_log(Mat &center_img){
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	phase(temp[0], temp[1], center_img);

	center_img += Scalar::all(1);
	log(center_img, center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//滤波器通用模板
//重新调整图片大小,类似matlab的paddsize，依据P*Q和傅里叶调整
resize_tpye* paddsize(const Mat &image_src) 
{  
   resize_tpye* result = (resize_tpye*)malloc(sizeof(resize_tpye));//调整以后的大小
   int rows = image_src.rows * 2;//height
   int cols = image_src.cols * 2;//weight

   int oph = getOptimalDFTSize(rows);
   int opw = getOptimalDFTSize(cols);
   result->size_rows = oph;
   result->size_cols = opw;
   return result;
}

resize_tpye* graph_resize(Mat &image_src) {
	//防止二维傅里叶变换周期性导致的缠绕
	//适应最佳傅里叶变换图像尺寸调整大小
	image_src.convertTo(image_src, CV_32FC1);
	resize_tpye* result=(resize_tpye* )malloc(sizeof(resize_tpye));//记录增加的像素，IDFT以后好减去
	
	int N=image_src.rows*2;//height
	int M=image_src.cols*2;//weight

	result->size_rows = image_src.rows;//减去height
	result->size_cols = image_src.cols;//减去weight

	copyMakeBorder(image_src, image_src, 0, N - image_src.rows, 0, M - image_src.cols,
		BORDER_CONSTANT, Scalar::all(0));

	///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	int oph = getOptimalDFTSize(image_src.rows);
	int opw = getOptimalDFTSize(image_src.cols);
	int resize_w = oph - image_src.rows;
	int resize_c = opw - image_src.cols;

	copyMakeBorder(image_src, image_src, 0, oph - image_src.rows, 0, opw - image_src.cols,
		BORDER_CONSTANT, Scalar::all(0));

	result->size_rows = result->size_rows + resize_w;//最后要缩减的总height
	result->size_cols = result->size_cols + resize_c;//weight
	
	return result;
}

//用图像矩阵来重新缩放函数的矩阵  没有什么用处
void filter_resize(Mat &image_src) {
	//防止二维傅里叶变换周期性导致的缠绕
	image_src.convertTo(image_src, CV_32FC1);
	int N = image_src.rows/2;//height
	int M = image_src.cols/2;//weight

	copyMakeBorder(image_src, image_src, N, N, M, M,
		BORDER_CONSTANT, Scalar::all(0));

	///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	int oph = getOptimalDFTSize(image_src.rows);
	int opw = getOptimalDFTSize(image_src.cols);

	int oph_N = (oph - image_src.rows) / 2;
	int opw_N = (opw - image_src.cols) / 2;
	copyMakeBorder(image_src, image_src, oph_N, oph_N, opw_N, opw_N,
		BORDER_CONSTANT, Scalar::all(0));
}

//图片剪切
void image_cut(Mat &src_image, resize_tpye* mat_resize) {
	int resize_row = src_image.rows - mat_resize->size_rows;
	int resize_col = src_image.cols - mat_resize->size_cols;
	Mat cat(src_image, Rect(0, 0, resize_col, resize_row));
	src_image=cat.clone();
}

//双通道复数A=复数A*复数B,A和B都必须是双通道
void complex_mul(const Mat& A_N,const Mat& B_N,Mat &des)
{Mat A;
 Mat B;
 A_N.convertTo(A, CV_32FC2);
 B_N.convertTo(B, CV_32FC2);
 Mat real= Mat::zeros(A.size(), CV_32F);
 Mat ima = Mat::zeros(A.size(), CV_32F);
 
 Mat vector_A[] = {Mat::zeros(A.size(),CV_32F),Mat::zeros(A.size(),CV_32F)};
 split(A,vector_A);
 
 Mat vector_B[] = { Mat::zeros(A.size(),CV_32F),Mat::zeros(A.size(),CV_32F) };
 split(B, vector_B);

 vector_A[0];//real
 vector_A[1];//ima

 vector_B[0];//real
 vector_B[0];//ima

 for(size_t i = 0; i < A.rows; i++)
    { for(size_t j = 0; j < A.cols; j++)
        { real.at<float>(i,j)= vector_A[0].at<float>(i,j)*vector_B[0].at<float>(i,j)-vector_A[1].at<float>(i,j)*vector_B[1].at<float>(i,j);
          ima.at<float>(i,j) = vector_A[0].at<float>(i,j)*vector_B[1].at<float>(i,j)+vector_A[1].at<float>(i,j)*vector_B[0].at<float>(i,j);
        }
    }
 
 vector_A[0]= real;//real
 vector_A[1]= ima;//ima
 merge(vector_A, 2, des);
};

//计算MatA*(-1)^(i+j)
void pow_i_j(Mat& A) {
	A.convertTo(A, CV_32F);
	for(size_t i = 0; i < A.rows; i++)
	  {  for(size_t j = 0; j < A.cols; j++)
	        {
		        A.at<float>(i,j) = A.at<float>(i, j)*pow(-1.0,i+j);
		    }
	  }
}

//通用的滤波_API
//mode==0::.muld点乘,对应直接频域上生成H(u,v),
//mode==1::complex_mul复数乘法,对应空间滤波器生成的虚数奇函数转换的h（u,v）
void filtering_Api(Mat &src_image, Mat &filter_image,int mode=0)
{    
	//1)调整大小，补0
	resize_tpye* image_info = graph_resize(src_image);
	//3)中心化移动
	pow_i_j(src_image);

	//2)计算F(U，V)求傅里叶变换
	Mat real_src_filter;
	Mat ima_src_filter;
	Mat src_image_dft = fast_dft(src_image, real_src_filter, ima_src_filter);
	//namedWindow("简单滤波women:", WINDOW_NORMAL)

	//4)生成滤波器图像，调整大小，中心在p/2，Q/2地方
	//这里滤波模块要求中心对称
	
	if (mode == 1) 
	{   //出入的h（x，y）=filter_image都必须是实数和奇函数才能保证傅里叶变换以后在频域上是虚奇函数
		filter_image.convertTo(filter_image,CV_32F);
		pow_i_j(filter_image);//移动到频域中心
		Mat real;
		Mat ima;

		//把滤波器转换到频域上
		Mat h_dft = fast_dft(filter_image, real, ima);
		amplitude_log(h_dft);
		h_dft.convertTo(h_dft, CV_8U);
		imshow("H(u,v)滤波器频谱图:", h_dft);
		
		pow_i_j(ima);//还原到空间域左上角

		//防止由于计算误差导致的real非零，赋值所有real=0
		for (int i = 0; i < real.rows; i++)
		{
			for (int j = 0; j < real.cols; j++)
			{
				real.at<float>(i, j) = 0.0;
			}
		}
		Mat filter_vector[] = {real,ima};
		merge(filter_vector, 2, filter_image);

		//这段注释可以帮助理解为什么需要乘以（-1）^(u+v)*F(U,V)
		//Mat real_N;
		//Mat ima_N;
		//Mat h_tepm[] = {real,ima};
		//merge(h_tepm,2,h);
		//fourior_inverser(h, real_N, ima_N);
		//divide(real_N,real_N.rows*real_N.cols,real_N);
		//real_N.convertTo(real_N, CV_32S);
		//ima_N.convertTo(ima_N,CV_32S);
		//Scalar ss = sum(ima_N);
		//cout<< ss[0]<<endl;
		//for (size_t i = 0; i < real_N.rows; i++)
		//{
		//  for (size_t j = 0; j <real_N.cols; j++)
		//	{
		//		if (abs(real_N.at<int>(i,j))>0) {
		//			cout << "row:" << i << "|col:" << j << ",value:="<< real_N.at<int>(i, j) << endl;
		//		}
		//	}
		//}
		//waitKey(0);
	
	}

	 //5)卷积相乘
	if(mode == 1)
	   complex_mul(src_image_dft,filter_image,src_image_dft);
	if(mode == 0)//传入的频域上的H(u，v)必须是偶函数，才能保证进行点乘运算，还原回空间域以后是一个实数
	  { //偶函数*real偶函数=real偶函数，偶函数*ima奇函数=ima奇函数
		//复制real为：：（real，ima）=（real，real）
		Mat filter_vector[] = {filter_image ,filter_image };
		merge(filter_vector, 2, filter_image);
		src_image_dft = src_image_dft.mul(filter_image);
	  }

	//6)逆变换
	fourior_inverser(src_image_dft, real_src_filter, ima_src_filter);
	cv::divide(real_src_filter, real_src_filter.rows*real_src_filter.cols, real_src_filter);

	//7)由于原来的图像pow_i_j一次，现在还原回去，pow_i_j不改变相角和频谱
	pow_i_j(real_src_filter);

	//8)裁剪图片
	image_cut(real_src_filter, image_info);
	real_src_filter.convertTo(real_src_filter,CV_8U);
	src_image = real_src_filter.clone();
}

//返回一个在频域上构造的滤波器
struct arg_ILPF {
	int D0_radius;//半径
	int rows;//长度
	int cols;//宽度
};

Mat set_filter_at_frespace(char * nameoffilter,void* arg)
{   Mat result;
	if(strcmp(nameoffilter, "ILPF") == 0)
	  {int  D0_radius = ((arg_ILPF*)arg)->D0_radius;//输入一个ILPF的D0的半径
	   //cout<<"半径是"<< ((arg_ILPF*)arg)->D0_radius <<endl;
	   int rows = ((arg_ILPF*)arg)->rows;
	   int cols= ((arg_ILPF*)arg)->cols;
       
	   int P_mide = rows / 2;
	   int Q_mide = cols / 2;

	   //定义内部函数
	   auto ifin = [](int x_rows,int y_cols,int P_mide,int  Q_mide,int D0_radius)->bool {
		   bool r = false;//包含在D0中
		   if((int)sqrt(pow(x_rows - P_mide, 2.0) + pow(y_cols - Q_mide, 2.0))<D0_radius)
		     { r = true;
		/*       cout << (int)sqrt(pow(x_rows - P_mide, 2.0) + pow(y_cols - Q_mide, 2.0)) <<"|"<<x_rows <<"|" <<y_cols << endl;
			   cout << P_mide << "|" << Q_mide << "|" << endl;
			   cout << D0_radius << endl;
			   cout << "----------------------------" << endl;*/
		     }
	       return r;
	   };

	   result=Mat::zeros(rows, cols,CV_32F);

	   for (size_t i = 0; i < rows; i++)
	   {
		   for (size_t j = 0; j < cols; j++)
		      {
			   if (ifin(i, j, P_mide, Q_mide, D0_radius) == true)
				   result.at<float>(i, j) = 1.0;
		      }
	   }
	   
	  }
	return result.clone();
}

//二、opencv测试函数
//ILPF_test
void filter_ILPF_test(int rato)
{
	 stringstream ss;
	 Mat a = imread("C:/Users/Administrator/Desktop/opencv/a.png", IMREAD_GRAYSCALE);
	 //imshow("原图",a);

	 resize_tpye* PQ=paddsize(a);
	 //cout << PQ->size_cols<< endl;

	 arg_ILPF * input_ILPE=(arg_ILPF*)malloc(sizeof(arg_ILPF));
	 input_ILPE->D0_radius = PQ->size_rows / rato;//宽度的3分之1为半径
	 input_ILPE->rows = PQ->size_rows;
	 input_ILPE->cols = PQ->size_cols;

	 Mat filter_image=set_filter_at_frespace("ILPF", input_ILPE);
	 ss<<rato;
	 string mark;
	 ss >> mark;
	 filtering_Api(a, filter_image,0);
	 imshow("a滤波后的图"+mark+":",a);
}

//一般输入进去的filter滤波器已经是中心化以后的了filter_need_center=false
void filtering(Mat &src_image, Mat &filter_image,bool filter_need_center=false)
{  
    //1)调整大小，补0
	resize_tpye* image_info = graph_resize(src_image);
    
	//2)计算F(U，V)求傅里叶变换
	Mat real_src_filter;
	Mat ima_src_filter;
	Mat src_image_dft = fast_dft(src_image, real_src_filter, ima_src_filter);
	//namedWindow("简单滤波women:", WINDOW_NORMAL)

	//3)中心化移动
	move_to_center(src_image_dft);

	//4)生成滤波器图像，调整大小，中心在p/2，Q/2地方
	//滤波器的处理一般在外面处理好再进函数
    if(filter_need_center ==true)
	   move_to_center(filter_image);//进来的

	//5)卷积相乘
    src_image_dft =src_image_dft.mul(filter_image);

	//6)逆变换
    fourior_inverser(src_image_dft, real_src_filter, ima_src_filter);
	divide(real_src_filter, real_src_filter.rows*real_src_filter.cols,real_src_filter);
	
	//7)返回中心
	for (int i = 0; i < real_src_filter.rows; i++)
	{
		for (int j = 0; j < real_src_filter.cols; j++)
		{
			real_src_filter.at<float>(i,j) = real_src_filter.at<float>(i, j)* pow(-1.0, i + j);
		}
	}
    
	//8)裁剪图片
    image_cut(real_src_filter, image_info);
    src_image = real_src_filter.clone();
}

//简单滤波器实验
void  simple_filter_test() {
	  Mat women = imread("C:/Users/Administrator/Desktop/lena.jpg", IMREAD_GRAYSCALE);
	  imshow("原图lena:", women);
	  //构造简单滤波器
	  Mat filter_image = Mat::ones(cv::Size(women.size()), CV_8UC1);
	  filter_image.data[0] = 0;
	  move_to_center(filter_image);
	  filter_resize(filter_image);

	  //画出滤波器
	  Mat filter_image_N=filter_image.clone();
	  namedWindow("简单滤波filter:", WINDOW_NORMAL);
	  filter_image_N = filter_image_N * 100.0;
	  filter_image_N.convertTo(filter_image_N, CV_8U);
	  imshow("简单滤波filter:", filter_image_N);
	 
	  Mat vector[] = { filter_image ,filter_image };
	  merge(vector,2,filter_image);
	  
	  //进行滤波操作
	  filtering(women, filter_image, false);
	  women.convertTo(women, CV_8U);
	  imshow("简单滤波lena:", women);
	  waitKey(0);
}

//house 实验
void house_test(Mat& image) {
	Mat real;
	Mat ima;
	Mat image_dft=fast_dft(image,real,ima);
	//cout<<image<<endl;
	amplitude_log(image_dft);
	image=image_dft.clone();
}

int opencv_DFT()
{
	Mat image = imread("C:/Users/Administrator/Desktop/I.png", IMREAD_GRAYSCALE);
	if (image.empty())
		return -1;

	imshow("src", image);
	image.convertTo(image, CV_32FC1);

	/////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	int oph = getOptimalDFTSize(image.rows);
	int opw = getOptimalDFTSize(image.cols);
	Mat padded;
	copyMakeBorder(image, padded, 0, oph - image.rows, 0, opw - image.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
	Mat complexI;
	merge(temp, 2, complexI);

	dft(complexI, complexI);
	split(complexI, temp);

	Mat amplitude, angle;
	//magnitude(temp[0], temp[1], amplitude);
	//phase(temp[0], temp[1], angle);
	cartToPolar(temp[0], temp[1], amplitude, angle);

	int cx = amplitude.cols / 2;
	int cy = amplitude.rows / 2;
	Mat q0(amplitude, Rect(0, 0, cx, cy));
	Mat q1(amplitude, Rect(cx, 0, cx, cy));
	Mat q2(amplitude, Rect(0, cy, cx, cy));
	Mat q3(amplitude, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	Mat amplitude_src;
	divide(amplitude, oph*opw, amplitude_src);

	imshow("amplitude_src", amplitude_src);

	amplitude += Scalar::all(1);
	log(amplitude, amplitude);
	normalize(amplitude, amplitude, 0, 255, NORM_MINMAX); //归一化 方便显示，和实际数据没有关系
	amplitude.convertTo(amplitude, CV_8U);
	imshow("amplitude", amplitude);

	Mat a0(angle, Rect(0, 0, cx, cy));
	Mat a1(angle, Rect(cx, 0, cx, cy));
	Mat a2(angle, Rect(0, cy, cx, cy));
	Mat a3(angle, Rect(cx, cy, cx, cy));

	Mat tmp_a;
	a0.copyTo(tmp_a);
	a3.copyTo(a0);
	tmp_a.copyTo(a3);

	a1.copyTo(tmp_a);
	a2.copyTo(a1);
	tmp_a.copyTo(a2);

	angle += Scalar::all(1);
	log(angle, angle);
	normalize(angle, angle, 0, 255, NORM_MINMAX); //归一化 方便显示，和实际数据没有关系
	angle.convertTo(angle, CV_8U);

	imshow("angle", angle);
	waitKey(0);
	return 1;
}

//频谱图和相位的结合
void fre_angle_graph_opencv() {
	Mat Lena1 = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	Mat real;
	Mat ima;
	Mat temp[] = { real,ima };
	//正傅里叶变换
	Mat complexI_fre = fast_dft(Lena1, real, ima);
	Mat complexI_angle = complexI_fre.clone();
	Mat complexI_only_fre = complexI_fre.clone();
	Mat complexI_only_angle = complexI_fre.clone();

	//频谱
	amplitude_log(complexI_fre);
	//move_to_center(complexI_fre);
	imshow("amplitude", complexI_fre);

	//相谱
	angle_log(complexI_angle);
	move_to_center(complexI_angle);
	imshow("angle", complexI_angle);

	//一、只保留频谱
	complexI_only_fre.convertTo(complexI_only_fre, CV_32FC2);
	Mat temp_only_fre[] = { Mat::zeros(complexI_only_fre.size(),CV_32FC1),Mat::zeros(complexI_only_fre.size(),CV_32FC1) };
	split(complexI_only_fre, temp_only_fre);//切分为实部和虚部
	magnitude(temp_only_fre[0], temp_only_fre[1], complexI_only_fre);
	complexI_only_fre = complexI_only_fre * (1.0 / sqrt(2.0));//去除相位，只保留频谱,每个相位只保留

	Mat vector[] = { complexI_only_fre,complexI_only_fre };
	merge(vector, 2, complexI_only_fre);
	Mat complexI_only_fre_N = fourior_inverser(complexI_only_fre, real, ima);
	divide(real, complexI_only_fre_N.rows*complexI_only_fre_N.cols, real);
	real.convertTo(real, CV_8U);
	move_to_center(real);
	//magnitude(real, ima, real);
	//normalize(real, real, 0, 255, NORM_MINMAX);
	imshow("只保留频谱", real);


	//二、只保留相位
	complexI_only_angle.convertTo(complexI_only_angle, CV_32FC2);
	Mat temp_only_angle[] = { Mat::zeros(complexI_only_angle.size(),CV_32FC1),Mat::zeros(complexI_only_angle.size(),CV_32FC1) };
	split(complexI_only_angle, temp_only_angle);//切分为实部和虚部

	Mat complexI_only_angle_mag;//计算频谱图，归一化所有傅里叶变换点
	magnitude(temp_only_angle[0], temp_only_angle[1], complexI_only_angle_mag);

	Mat vector_angle[] = { complexI_only_angle_mag ,complexI_only_angle_mag };
	merge(vector_angle, 2, complexI_only_angle_mag);
	//divide(complexI_only_angle, complexI_only_angle_mag, complexI_only_angle, 1.0);
	complexI_only_angle = complexI_only_angle / complexI_only_angle_mag;

	Mat complexI_only_angle_N = fourior_inverser(complexI_only_angle, real, ima);
	//divide(real, complexI_only_angle_N.rows*complexI_only_angle_N.cols, real);//
	magnitude(real, ima, real);//分离通道，主要获取0通道
	normalize(real, real, 0, 1, NORM_MINMAX);//归
	//real.convertTo(real, CV_8U);
	imshow("只保留相位", real);

	//三、保留女人图的相位，使用I图频谱||保留女人图的频谱，使用I图的相位
	Mat women = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	Mat I = imread("C:/Users/Administrator/Desktop/I.png", IMREAD_GRAYSCALE);
	float col_rato = (float)women.cols / (float)I.cols;
	float row_rato = (float)women.rows / (float)I.rows;
	imshow("原图I", I);
	imshow("原图women", women);
	resize(I, I, cv::Size(), col_rato, row_rato);
	imshow("缩放I", I);

	I.convertTo(I, CV_32FC1);
	women.convertTo(women, CV_32FC1);

	Mat realI;
	Mat imaI;
	women = fast_dft(women, real, ima);
	I = fast_dft(I, realI, imaI);

	//1）保留女人图的相位，使用I图频谱
	complexI_only_angle = women.clone();//使用women图的相位
	Mat temp_wangle_Ifre[] = { Mat::zeros(complexI_only_angle.size(),CV_32FC1),Mat::zeros(complexI_only_angle.size(),CV_32FC1) };
	split(complexI_only_angle, temp_wangle_Ifre);//切分为实部和虚部

	//计算频谱图，归一化所有傅里叶变换点
	magnitude(temp_wangle_Ifre[0], temp_wangle_Ifre[1], complexI_only_angle_mag);

	Mat vector_wangle_Ifre[] = { complexI_only_angle_mag ,complexI_only_angle_mag };
	merge(vector_wangle_Ifre, 2, complexI_only_angle_mag);

	//--------------------------计算I的频谱-------------------------------
	Mat I_fre = I.clone();//使用I图的频谱
	Mat temp_Ifre[] = { Mat::zeros(I_fre.size(),CV_32FC1),Mat::zeros(I_fre.size(),CV_32FC1) };
	split(I_fre, temp_Ifre);//切分为实部和虚部
	Mat I_fre_fre;
	magnitude(temp_Ifre[0], temp_Ifre[1], I_fre_fre);
	Mat I_fre_fre_vector[] = { I_fre_fre, I_fre_fre };
	merge(I_fre_fre_vector, 2, I_fre_fre);
	//--------------------------结束计算I的频谱---------------------------

	//保留女人图的相位，使用I图的频谱
	complexI_only_angle = (complexI_only_angle / complexI_only_angle_mag);
	complexI_only_angle = I_fre_fre.mul(complexI_only_angle);

	complexI_only_angle_N = fourior_inverser(complexI_only_angle, real, ima);
	//divide(real, complexI_only_angle_N.rows*complexI_only_angle_N.cols, real);//
	magnitude(real, ima, real);//分离通道，主要获取0通道
	normalize(real, real, 0, 1, NORM_MINMAX);//归
	//real.convertTo(real, CV_8U);
	imshow("保留女人图的相位，使用I图频谱", real);


	//2）保留女人图的频谱，使用I图相位
	complexI_only_angle = I.clone();//使用women图的相位
	Mat temp_wangle_wfre[] = { Mat::zeros(complexI_only_angle.size(),CV_32FC1),Mat::zeros(complexI_only_angle.size(),CV_32FC1) };
	split(complexI_only_angle, temp_wangle_wfre);//切分为实部和虚部

	//计算频谱图，归一化所有傅里叶变换点
	magnitude(temp_wangle_wfre[0], temp_wangle_wfre[1], complexI_only_angle_mag);

	Mat vector_Iangle_Wfre[] = { complexI_only_angle_mag ,complexI_only_angle_mag };
	merge(vector_Iangle_Wfre, 2, complexI_only_angle_mag);

	//--------------------------计算I的频谱-------------------------------
	I_fre = women.clone();//使用I图的频谱
	Mat temp_wfre[] = { Mat::zeros(I_fre.size(),CV_32FC1),Mat::zeros(I_fre.size(),CV_32FC1) };
	split(I_fre, temp_wfre);//切分为实部和虚部
	I_fre_fre;
	magnitude(temp_wfre[0], temp_wfre[1], I_fre_fre);
	Mat I_fre_wre_vector[] = { I_fre_fre, I_fre_fre };
	merge(I_fre_wre_vector, 2, I_fre_fre);
	//--------------------------结束计算I的频谱---------------------------

	//保留女人图的相位，使用I图的频谱
	complexI_only_angle = (complexI_only_angle / complexI_only_angle_mag);
	complexI_only_angle = I_fre_fre.mul(complexI_only_angle);

	complexI_only_angle_N = fourior_inverser(complexI_only_angle, real, ima);
	//divide(real, complexI_only_angle_N.rows*complexI_only_angle_N.cols, real);//
	magnitude(real, ima, real);//分离通道，主要获取0通道
	normalize(real, real, 0, 1, NORM_MINMAX);//归
	//real.convertTo(real, CV_8U);
	imshow("保留女人图的频谱，使用I图相位", real);
}