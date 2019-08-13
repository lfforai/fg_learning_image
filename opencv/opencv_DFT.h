#pragma once
#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
//系统包
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"

using namespace std;
using namespace cv;
struct resize_tpye {
	int size_rows = 0;//记录傅里叶最优化增加的0元素
	int size_cols = 0;
};

int opencv_DFT();
//实验
void fre_angle_graph_opencv();
void house_test(Mat& image);
void image_cut(Mat &src_image, resize_tpye* mat_resize);
void filter_ILPF_test(int rato);
void filter_test();
void filter_ILPF_bell(int rato);
void filter_BLPF_test(int rato, int n);


struct Met_oe_info {
	int real_odd_or_even; //奇数=1，偶数=2,非奇数非偶=0,real
	int ima_odd_or_even;//奇数=1，偶数=2,非奇数非偶=0，ima
	int channls;//如果是单通道，结果默认放在real
	void print() {
		cout << "channls:"<< channls << endl;
		cout << "real:"<< real_odd_or_even << endl;
		if (channls>1)
		   cout << "ima:" << ima_odd_or_even << endl;
	}
};

//通用api
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
Mat fourior_inverser(Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
Mat fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
void move_to_center(Mat &center_img);
void amplitude_log(Mat &center_img);
void amplitude_common(Mat &center_img);
void angle_common(Mat &center_img);
void angle_log(Mat &center_img);
resize_tpye* graph_resize(Mat &image_src);
Met_oe_info * Mat_is_odd_or_even(const Mat image);
Mat amplitude_common_from_iamge(Mat &image);
Mat amplitude_log_from_iamge(Mat &image);
Mat image2_copy(const Mat& big, const Mat& less);
void fre2space_show(char * namefilter);
void image_show(const Mat& image, float rato, const char * c = "图像");
void image_show(const Mat& image, float rows, float cols, const char * c);
//