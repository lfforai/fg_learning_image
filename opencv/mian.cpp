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
using namespace image_scale0;
using namespace image_scale1;
using namespace image_scale2;

int main()
{
	//image_scale0::image_scale0();
	//image_scale1::image_scale1();
	//image_scale2::image_scale2();
	//cuffttest("C:/Users/Administrator/Desktop/I.png");

	//��ѧʵ��
	//atan_cpu_test();
	//cufft_math_test("C:/Users/Administrator/Desktop/I.png",0);

	Mat lena=image_rotate_point_GPU("C:/Users/Administrator/Desktop/I.png",Mat::ones(2,2,0),0);
	
	waitKey(0);
}

