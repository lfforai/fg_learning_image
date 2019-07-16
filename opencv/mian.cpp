#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
#include "opencv_DFT.h"
#include "math_cuda.cuh"

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
	Mat Lena1 = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	Mat real;
	Mat ima;
	Mat temp[] = { real,ima };
	//正傅里叶变换
	Mat complexI_fre = fast_dft(Lena1, real, ima);
	Mat complexI_angle =complexI_fre.clone();
	Mat complexI_only_fre = complexI_fre.clone();
	Mat complexI_only_angle = complexI_fre.clone();
	
	//频谱
	amplitude_log(complexI_fre);
	move_to_center(complexI_fre);
	imshow("amplitude", complexI_fre);

	//相谱
	angle_log(complexI_angle);
	move_to_center(complexI_angle);
	imshow("angle", complexI_angle);

	//一、只保留频谱
	complexI_only_fre.convertTo(complexI_only_fre, CV_32FC2);
	Mat temp_only_fre[] ={ Mat::zeros(complexI_only_fre.size(),CV_32FC1),Mat::zeros(complexI_only_fre.size(),CV_32FC1) };
	split(complexI_only_fre, temp_only_fre);//切分为实部和虚部
	magnitude(temp_only_fre[0],temp_only_fre[1],complexI_only_fre);
	complexI_only_fre =complexI_only_fre * (1.0/sqrt(2.0));//去除相位，只保留频谱,每个相位只保留
	
	Mat vector[] = {complexI_only_fre,complexI_only_fre};
	merge(vector,2,complexI_only_fre);
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
	
	Mat vector_angle[] = {complexI_only_angle_mag ,complexI_only_angle_mag};
	merge(vector_angle,2,complexI_only_angle_mag);
	//divide(complexI_only_angle, complexI_only_angle_mag, complexI_only_angle, 1.0);
	complexI_only_angle =complexI_only_angle/complexI_only_angle_mag;
	
	Mat complexI_only_angle_N = fourior_inverser(complexI_only_angle, real, ima);
	//divide(real, complexI_only_angle_N.rows*complexI_only_angle_N.cols, real);//
	magnitude(real,ima,real);//分离通道，主要获取0通道
	normalize(real,real, 0, 1, NORM_MINMAX);//归
	//real.convertTo(real, CV_8U);
	imshow("只保留相位", real);

	//三、保留女人图的相位，使用I图频谱||保留女人图的频谱，使用I图的相位
	Mat women = imread("C:/Users/Administrator/Desktop/old.png", IMREAD_GRAYSCALE);
	Mat I = imread("C:/Users/Administrator/Desktop/I.png", IMREAD_GRAYSCALE);
	float col_rato = (float)women.cols / (float)I.cols;
	float row_rato = (float)women.rows / (float)I.rows;
	imshow("原图I", I);
	imshow("原图women", women);
	resize(I,I,cv::Size(),col_rato,row_rato);
	imshow("缩放I", I);
	
	I.convertTo(I, CV_32FC1);
	women.convertTo(women, CV_32FC1);

	Mat realI;
	Mat imaI;
	women=fast_dft(women, real, ima);
	I=fast_dft(I, realI, imaI);

	//1）保留女人图的相位，使用I图频谱
	complexI_only_angle = women.clone();//使用women图的相位
	Mat temp_wangle_Ifre[]={ Mat::zeros(complexI_only_angle.size(),CV_32FC1),Mat::zeros(complexI_only_angle.size(),CV_32FC1)};
	split(complexI_only_angle, temp_wangle_Ifre);//切分为实部和虚部

    //计算频谱图，归一化所有傅里叶变换点
	magnitude(temp_wangle_Ifre[0], temp_wangle_Ifre[1], complexI_only_angle_mag);

	Mat vector_wangle_Ifre[] = {complexI_only_angle_mag ,complexI_only_angle_mag};
	merge(vector_wangle_Ifre, 2, complexI_only_angle_mag);
	
	//--------------------------计算I的频谱-------------------------------
	Mat I_fre = I.clone();//使用I图的频谱
	Mat temp_Ifre[] = { Mat::zeros(I_fre.size(),CV_32FC1),Mat::zeros(I_fre.size(),CV_32FC1) };
	split(I_fre, temp_Ifre);//切分为实部和虚部
	Mat I_fre_fre;
	magnitude(temp_Ifre[0], temp_Ifre[1], I_fre_fre);
	Mat I_fre_fre_vector[]={I_fre_fre, I_fre_fre};
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

int main()
{
	fre_angle_graph();
	waitKey(0);
}

