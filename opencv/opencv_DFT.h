#pragma once
#include "cufft.cuh"        //����Ҷ�˲�ʽ��
#include "image_scale.cuh"  //ͼ������ʵ��
//ϵͳ��
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"

using namespace std;
using namespace cv;

int opencv_DFT();
//ʵ��
void fre_angle_graph_opencv();
//ͨ��api
Mat fourior_inverser(Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
Mat fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img);
void move_to_center(Mat &center_img);
void amplitude_log(Mat &center_img);
void amplitude_common(Mat &center_img);
void angle_common(Mat &center_img);
void angle_log(Mat &center_img);
void  filter_test();