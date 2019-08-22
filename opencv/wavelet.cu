#include "wavelet.cuh"

//#pragma comment(lib, "python37.lib")
//mode=0 up,mode=1 down; row_or_col=0 row,row_or_col=1 col
Mat up_down_sample(Mat& image,int mode,int row_or_col) 
{   Mat result;
	if(mode == 0)//下采样
	 { if(row_or_col==0)//row
	   resize(image, result, Size(image.cols,(int)(image.rows/2.0)), 0, 0, INTER_NEAREST);
	   if(row_or_col==1)//col
	   resize(image, result, Size((int)(image.cols/2.0), image.rows), 0, 0, INTER_NEAREST);
	 }
	 else//上采样
	 {
	  if(row_or_col == 0)//row
		{
			Mat new_mat = Mat::zeros(image.rows*2, image.cols, CV_8U);
			int M = image.rows;
			int N = image.cols;
			for(size_t i = 0; i < M; i++)
			  {
				for(size_t j = 0; j < N; j++)
				  {
					new_mat.at<uchar>(i * 2+1, j) = image.at<uchar>(i, j);
				  }
			  }
			result=new_mat.clone();
		}
		else //col
	   {
			Mat new_mat = Mat::zeros(image.rows, image.cols*2, CV_8U);
			int M = image.rows;
			int N = image.cols;
			for (size_t i = 0; i < M; i++)
			{
				for (size_t j = 0; j < N; j++)
				{
					new_mat.at<uchar>(i, j * 2+1) = image.at<uchar>(i, j);
				}
			}
			result = new_mat.clone();
		}
	 }
	return result.clone();
}

void  up_down_sample_test() {

	Mat A = Mat::zeros(3, 3, CV_8U);
	A.at<uchar>(0, 0) = 1;
	A.at<uchar>(1, 0) = 1;
	A.at<uchar>(2, 0) = 1;
	A.at<uchar>(0, 1) = 2;
	A.at<uchar>(1, 1) = 2;
	A.at<uchar>(2, 1) = 2;
	A.at<uchar>(0, 2) = 3;
	A.at<uchar>(1, 2) = 3;
	A.at<uchar>(2, 2) = 3;

	Mat B = Mat::zeros(9, 1, CV_8U);
	B.at<uchar>(0, 0) = 1;
	B.at<uchar>(1, 0) = 2;
	B.at<uchar>(2, 0) = 3;
	B.at<uchar>(3, 0) = 4;
	B.at<uchar>(4, 0) = 5;
	B.at<uchar>(5, 0) = 6;
	B.at<uchar>(6, 0) = 7;
	B.at<uchar>(7, 0) = 8;
	B.at<uchar>(8, 0) = 9;

	Mat up_sample=up_down_sample(A,1,1);
	cout << up_sample << endl;
	Mat result = up_down_sample(up_sample, 0, 1);
	cout<< result <<endl;

	up_sample = up_down_sample(B, 1, 0);
	cout << up_sample << endl;
	result = up_down_sample(up_sample, 0, 0);
	cout << result << endl;
}


void wavelet_test(){
	Py_Initialize(); /*初始化python解释器,告诉编译器要用的p
	up_down_sample_test();ython编译器*/
	PyRun_SimpleString("import numpy as np"); /*调用python文件*/
	Py_Finalize(); /*结束python解释器，释放资源*/
	system("pause");
}

