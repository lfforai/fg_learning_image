#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
#include "opencv_DFT.h"
#include "math_cuda.cuh"
#include "opencv_chapter3.h"

//系统包
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"

using namespace std;
using namespace cv;

using namespace image_scale0;
using namespace image_scale1;
using namespace image_scale2;

//一、图像放缩实验
void graph_scale_test()
{
	image_scale0::image_scale0();
	image_scale1::image_scale1();
	image_scale2::image_scale2();
}

//二、傅里叶实验
void cuttf_test() {
	cuffttest("C:/Users/Administrator/Desktop/I.png");
}

//三、频谱_相谱实验
void fre_angel_graph_test() {
	//原始图
	Mat lena1 = imread("C:/Users/Administrator/Desktop/I.png");
	cufftComplex* data1 = cufft_fun("", lena1, 0, 1, 0, 1);
	Mat output1 = fre_spectrum(data1, lena1.cols, lena1.rows, 1);
	imshow("fg的log频谱图：", output1);

	Mat  angle_mat = angle_spectrum(data1, lena1.cols, lena1.rows);
	imshow("fg的log相谱图：", angle_mat);
	cudaFree(data1);

	//图像旋转
	Mat lena = image_rotate_point_GPU("C:/Users/Administrator/Desktop/I.png", Mat::ones(2, 2, 0), 0);
	imshow("fg的旋转图：", lena);

	cufftComplex* data_rotate = cufft_fun("", lena, 0, 1, 0, 0);
	Mat fre_rotate_mat = fre_spectrum(data_rotate, lena.cols, lena.rows, 1);
	imshow("fg的log旋转的频谱图：", fre_rotate_mat);

	Mat  angle_rotate_mat = angle_spectrum(data_rotate, lena1.cols, lena1.rows);
	imshow("fg的log旋转的相谱图：", angle_rotate_mat);
	cudaFree(data_rotate);

	//坐标平移
	Mat lena_move = image_move_point_GPU("C:/Users/Administrator/Desktop/I.png", Mat::ones(2, 2, 0), 0, 140, -50);
	imshow("fg的坐标点移动图：", lena_move);

	cufftComplex* data_move = cufft_fun("", lena_move, 0, 1, 0, 0);
	Mat  fre_move_mat = fre_spectrum(data_move, lena.cols, lena.rows, 1);
	imshow("fg的log平移后的频谱图：", fre_move_mat);

	Mat  angle_move_mat = angle_spectrum(data_move, lena1.cols, lena1.rows);
	imshow("fg的log平移后的相谱图：", angle_move_mat);
	cudaFree(data_move);
	
	//全opencv版本
	opencv_DFT();
	waitKey(0);
}

//四、cufft的正傅里叶变换和负傅里叶变换能还原图像，opencv的idft也能还原，计算准确
void _cuttf_idft() 
{
	Mat Lena = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	imshow("原图", Lena);
	int oph = getOptimalDFTSize(Lena.rows);
	int opw = getOptimalDFTSize(Lena.cols);
	Mat padded;
	copyMakeBorder(Lena, padded, 0, oph - Lena.rows, 0, opw - Lena.cols,
		BORDER_CONSTANT, Scalar::all(0));
	hy_fun(Lena);

	Mat Lena1 = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	Mat real;
	Mat ima;
	Mat temp[] = { real,ima };
	//正傅里叶变换
	Mat complexI = fast_dft(Lena1, real, ima);
	//反傅里叶变换
	complexI = fourior_inverser(complexI, real, ima);
	divide(real, complexI.rows*complexI.cols, real);
	real.convertTo(real, CV_8U);
	//magnitude(real, ima, real);
	//normalize(real, real, 0, 255, NORM_MINMAX);
	imshow("idft还原图", real);
}

//五、相位_频谱的相互保留
void fre_angle_graph() {
	fre_angle_graph_opencv();
}
//六、第三章的各种测试
void chapter3() {
	chapter3_test();
}


//七、从空间滤波到频率滤波的等价性
void charter4() {
	//参考微博https://blog.csdn.net/lvsehaiyang1993/article/details/80876712
	//例4.15从模板到频域滤波器
	Mat lena = imread("C:/Users/Administrator/Desktop/opencv/house.png", IMREAD_GRAYSCALE);
	Mat z = Mat::zeros(316, 316, CV_32F);
	resize(lena, lena, z.size(), INTER_CUBIC);
	Mat dft_lena = lena.clone();//频域滤波用
	Mat dft_lena_filter_space = lena.clone();//空间滤波用图
	imshow("原图house:", lena);
	house_test(lena);
	move_to_center(lena);
	imshow("log频谱house:", lena);

	//一、频域滤波的大小  
	//扩展为P*Q,
	/*copyMakeBorder(dft_lena, dft_lena,1, 1, 1,1,
		BORDER_CONSTANT, Scalar::all(0));*/
	resize_tpye* re = graph_resize(dft_lena);
	dft_lena.convertTo(dft_lena, CV_8U);
	imshow("原图P*Q扩展以后图:", dft_lena);
	cout << "需要缩减的大小：" << re->size_cols << "|" << re->size_rows << endl;
	Mat real;
	Mat ima;

	Mat h = Mat::zeros(dft_lena.size(), CV_32F);
	cout << "必须是偶数，并且和原图频域上一样大" << h.size() << endl;//必须是个偶数
	int r_m = (int)(h.rows / 2.0) - 1;
	int c_n = (int)(h.cols / 2.0) - 1;

	//h.at<float>(1, 0) = 2;  //必须放在这个位置傅里叶变换以后才是虚奇函数
	//h.at<float>(1, 1) = 1;
	//h.at<float>(1, h.cols - 1) = 1;
	//h.at<float>(h.rows - 1, 0) = -2;
	//h.at<float>(h.rows - 1, 1) = -1;
	//h.at<float>(h.rows - 1, h.cols - 1) = -1;
	 
	h.at<float>(r_m, c_n) =  1;  //必须放在这个位置傅里叶变换以后才是虚奇函数
	h.at<float>(r_m, c_n + 1) = 2;
	h.at<float>(r_m, c_n + 2) = 1;
	h.at<float>(r_m + 1, c_n) = 0;
	h.at<float>(r_m + 1, c_n + 1) = 0;
	h.at<float>(r_m + 1, c_n + 2) = 0;
	h.at<float>(r_m + 2, c_n) = -1;
	h.at<float>(r_m + 2, c_n + 1) = -2;
	h.at<float>(r_m + 2, c_n + 2) = -1;

	for (int i = 0; i < h.rows; i++)
	{
		for (int j = 0; j < h.cols; j++)
		{
			h.at<float>(i, j) = h.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	//把滤波器转换到频域上
	Mat h_dft = fast_dft(h, real, ima);
	amplitude_log(h_dft);
	h_dft.convertTo(h_dft, CV_8U);
	imshow("滤波器频谱图:", h_dft);
	for (int i = 0; i < ima.rows; i++)
	{
		for (int j = 0; j < ima.cols; j++)
		{
			ima.at<float>(i, j) = ima.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	for (int i = 0; i < real.rows; i++)
	{
		for (int j = 0; j < real.cols; j++)
		{
			real.at<float>(i, j) = 0.0;
		}
	}

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

	//二、正式滤波过程
	//1)调整大小，补0
	//2)计算F(U，V)求傅里叶变换
	Mat dft_lena_filter;
	dft_lena.convertTo(dft_lena_filter, CV_32F);
	for (int i = 0; i < dft_lena_filter.rows; i++)
	{
		for (int j = 0; j < dft_lena_filter.cols; j++)
		{
			dft_lena_filter.at<float>(i, j) = dft_lena_filter.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	Mat real_src_filter;
	Mat ima_src_filter;
	Mat src_image_dft = fast_dft(dft_lena_filter, real_src_filter, ima_src_filter);

	//3)中心化移动
	//move_to_center(src_image_dft);

	//4)生成滤波器图像，调整大小，中心在p/2，Q/2地方

	//5)卷积相乘
	//src_image_dft = src_image_dft.mul(h_dft);
	Mat vector[] = { Mat::zeros(real.size(),CV_32F),Mat::zeros(ima.size(),CV_32F) };
	split(src_image_dft, vector);
	Mat image_real = vector[0];
	Mat image_ima = vector[1];
	real = Mat::zeros(real.size(), CV_32F);

	for (size_t i = 0; i < image_real.rows; i++)
	{
		for (size_t j = 0; j < image_real.cols; j++)
		{   //real 全部为零，所有才有下面简化版复数（a+bi）*（c+di)
			image_real.at<float>(i, j) = -image_ima.at<float>(i, j)*ima.at<float>(i, j);
			image_ima.at<float>(i, j) = image_real.at<float>(i, j)*ima.at<float>(i, j);
			//还原回原函数
			//image_real.at<float>(i, j) = image_real.at<float>(i, j);
			//image_ima.at<float>(i, j) = image_ima.at<float>(i, j);
		}
	}

	vector[0] = image_real;
	vector[1] = image_ima;
	merge(vector, 2, src_image_dft);

	//6)逆变换
	fourior_inverser(src_image_dft, real_src_filter, ima_src_filter);
	divide(real_src_filter, real_src_filter.rows*real_src_filter.cols, real_src_filter);
	divide(ima_src_filter, ima_src_filter.rows*ima_src_filter.cols, ima_src_filter);
	//cout<< real_src_filter <<endl;
	//cout<<ima_src_filter<<endl;

	//7)返回中心
	for (int i = 0; i < real_src_filter.rows; i++)
	{
		for (int j = 0; j < real_src_filter.cols; j++)
		{
			real_src_filter.at<float>(i, j) = real_src_filter.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	//demarcate(real_src_filter);
	real_src_filter.convertTo(real_src_filter, CV_8U);
	image_cut(real_src_filter, re);
	//8)裁剪图片
	imshow("滤波后的图:", real_src_filter);

	Laplace_cuda(dft_lena_filter_space, 3, 1);
	dft_lena_filter_space.convertTo(dft_lena_filter_space, CV_8U);

	//demarcate(dft_lena_filter_space);
	imshow("空间滤波后的图:", dft_lena_filter_space);
	waitKey(0);
}

int main()
{
	charter4();

}

