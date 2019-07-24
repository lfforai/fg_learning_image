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

//通用api
Mat fourior_inverser(Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
Mat fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
void move_to_center(Mat &center_img);
void amplitude_log(Mat &center_img);
void amplitude_common(Mat &center_img);
void angle_common(Mat &center_img);
void angle_log(Mat &center_img);
resize_tpye* graph_resize(Mat &image_src);
void filter_test();