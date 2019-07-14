#include "cufft.cuh"        //傅里叶滤波式样
#include "image_scale.cuh"  //图形缩放实验
//系统包
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace image_scale0;
using namespace image_scale1;
using namespace image_scale2;

int main()
{
	
	//image_scale0::image_scale0();
	//image_scale1::image_scale1();
	//image_scale2::image_scale2();
	cuffttest("C:/Users/Administrator/Desktop/I.png");
}

