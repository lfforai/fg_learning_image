#include "opencv_DWT.h"

//例7.2 4带宽子带编码
void exchange(float* src,float* des,int len)
{ for (size_t i = 0; i < len; i++)
	{
		des[len-1-i] = src[i];
	}
}

void pow_i(float* src,int len) {
	for (size_t i = 0; i < len; i++)
	{
		src[i] = src[i]*pow(-1.0,i+1);
	}
}

//len is length of h,row_or_cols:row =0,cols=1
Mat FRI(Mat& image_N,float* h,int len,int row_or_cols)
{
	Mat result=Mat::zeros(image_N.size(),CV_32F);
	Mat image = image_N.clone();
	image.convertTo(image, CV_32F);
	//按行
	if (row_or_cols == 0) {
		int rows = image.rows;
		int cols = image.cols;
		for (size_t j = 0; j < cols;j++)
		{ 
			for (size_t i = 0; i < rows; i++) {		
				//卷积
				float sum = 0;
				for (size_t w = 0; w <len; w++)
				    {     //按行展开
					  if (i + w < rows)
						 sum = sum + image.at<float>(i + w, j)*h[w];
					      //超过边界等于0
				    }
				result.at<float>(i, j) = sum;
			}
		}
		cv::resize(result, result, cv::Size(result.rows/2, result.cols),0,0,INTER_NEAREST);
	}

	//按列
	if (row_or_cols == 1) {
		int rows = image.rows;
		int cols = image.cols;

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++) {
				//卷积
				float sum = 0;
				for (size_t w = 0; w < len; w++)
				{   //按行展开
					if (j + w < cols)
						sum = sum + image.at<float>(i,j+w)*h[w];
					//超过边界等于0
				}
				result.at<float>(i, j) = sum;
			}
		}
		cv::resize(result, result, cv::Size(result.rows, result.cols/2), 0, 0, INTER_NEAREST);
	}
	return result.clone();
}

void base_code(char *path, float* h, int len) {
	Mat image = imread(path, IMREAD_GRAYSCALE);
	image_show(image, 1, "原图：");
	//waitKey(0);
	Mat image_f = image.clone();
	Mat image_s = image.clone();

	float * h_0 = (float *)malloc(len*sizeof(float));
	exchange(h, h_0, len);

	float * h_1 = (float *)malloc(len*sizeof(float));
	memcpy(h_1, h, len * sizeof(float));
	//exchange(h_1, h_0, len);
	pow_i(h_1, len);

	Mat h0 = FRI(image_f, h_0, 8, 0);
	//image_show(h0, 1, "h0：");
	Mat h1= FRI(image_s, h_1, 8, 0);
	//image_show(h1, 1, "h1：");

	Mat h0_h0 = FRI(h0, h_0, 8, 1);
	Mat h0_h1 = FRI(h0, h_1, 8, 1);
	
	Mat h1_h0 = FRI(h1, h_0, 8, 1);
	Mat h1_h1 = FRI(h1, h_1, 8, 1);

	resize(h0_h0,h0_h0,image.size());
	image_show(h0_h0, 1, "近似h0_h0：");
	resize(h0_h1, h0_h1, image.size());
	image_show(h0_h1, 1, "垂直h0_h1：");
	resize(h1_h0, h1_h0, image.size());
	image_show(h1_h0, 1, "横向h1_h0：");
	resize(h1_h1, h1_h1, image.size());
	image_show(h1_h1, 1, "对角h1_h1：");
}

