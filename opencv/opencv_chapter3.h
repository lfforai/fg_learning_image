#pragma once
#include "cufft.cuh"        //����Ҷ�˲�ʽ��
#include "image_scale.cuh"  //ͼ������ʵ��
#include  "image_rotate.cuh"
//ϵͳ��
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"
#include<time.h>

void graph_inverse(Mat& image_graph);
void chapter3_test();
cv::Mat getImageofHistogram(const cv::Mat &hist, int zoom);//����ֱ��ͼ
void Histogram(Mat& image);