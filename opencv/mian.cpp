#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
#include "opencv_DFT.h"
#include "math_cuda.cuh"
#include "opencv_chapter3.h"
#include "opencv_chapter4.h"
#include "opencv_DWT.h"
#include "morphology.cuh"
#include "opencv_chapter10.cuh"

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

//七、第四章的各种测试
void charter4() {
	chapter4();
}

//八、第七章的各种测试
void charter7() {
	float h[8] = { 0.23037781,0.714846577,0.6308076,-0.02798376,-0.1870341,0.03084138,0.03288301,-0.01059740 };
	base_code("C:/Users/Administrator/Desktop/opencv/wall.jpg", h, 8);
}

//九、第九章的各种测试
void charter9() {

}

int main()
{
	chapter10_test();
 //charter7();
  //morphology_test(12,10,1);
  //morphology_test(4, 4,1);
 //morphology_test(15, 15,1);
 //chapter10();
 waitKey(0);
}

