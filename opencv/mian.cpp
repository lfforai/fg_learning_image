#include "cufft.cuh"        //����Ҷ�˲�ʽ��
#include "image_scale.cuh"  //ͼ������ʵ��
//ϵͳ��
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

