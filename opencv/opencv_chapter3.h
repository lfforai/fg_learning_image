#pragma once
#include "cufft.cuh"        //����Ҷ�˲�ʽ��
#include "image_scale.cuh"  //ͼ������ʵ��
//ϵͳ��
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "math_cuda.cuh"
#include "image_rotate.cuh"

void graph_inverse(Mat& image_graph);
void chapter3_test();