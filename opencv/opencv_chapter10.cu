#include "opencv_chapter10.cuh"

//如果是init=0构造Hough_mat,init=1
Mat Hough_gpu(Mat& image_N)//Hough_mat必须是整数
{  
	Mat image = image_N.clone();
	image.convertTo(image, CV_32F);

	int angle_min = -90;//横向
	int angle_max = 90;

	int P_min = (int)-sqrt(pow(image.rows, 2.0) + pow(image.cols, 2.0));//纵向
	int P_max = (-1.0)*P_min;

	int p_len = P_max+1;
	int a_len = angle_max+1;

	Mat Hough_mat = Mat::zeros(2 * P_max + 1, 2 * angle_max + 1, CV_32S);


	int M = image.rows;
	int N = image.cols;
		
	for (int i = 0; i <M; i++)
		{  for (int j = 0; j <N; j++)
			{     
		    if(image.at<float>(i, j) > 0)//只处理非背景点
			    {
				for(int ang = angle_min; ang < angle_max + 1; ang++)
					{  
					int value=(int)(i*cos(ang*3.1415926 / 180.0) + j * sin(ang*3.1415926 / 180.0));
					//cout<<"value:"<<value<<"ang:"<<ang<<"|"<< value + p_len - 1<<"|"<< ang + a_len - 1 <<endl;
					Hough_mat.at<int>(value+p_len-1, ang+a_len-1) = Hough_mat.at<int>(value + p_len-1, ang + a_len-1)+1;
					} 
				}
			}
		}
	return Hough_mat.clone();
}


Mat Fill_Vertical(Mat& g ,int k) 
{
	Mat result_N = Mat::zeros(g.size(), CV_8U);
	int N = g.cols;
	int M = g.rows;

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			if (j < M - k)
			{
				if (g.at<uchar>(j, i) == 255)
				{
					result_N.at<uchar>(j, i) = 255;
					int k_n = 1;
					while (k_n < k)
					{
						if (g.at<uchar>(j + k_n, i) == 255)
							break;
						else
							k_n = k_n + 1;
					}

					if (k_n > 1 && g.at<uchar>(j + k_n, i) == 255 && k_n < k)
					{
						for (size_t n = 1; n <= k_n; n++)
						{
							result_N.at<uchar>(j + (int)n, i) = 255;
						}
					}
				}
			}
			else {

				result_N.at<uchar>(j, i) = g.at<uchar>(j, i);
			}
		}
	}
	return result_N.clone();
}

void Hough_test() 
{
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/air.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	image_show(Lena, 1, "原图");
  
	//一、先高通滤波，后拉普拉斯变换
	f_screem<float>* filter_G = set_f<float>(sf_mode::Gauss25_N);
	Mat GSmat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(GSmat, 0.4, "高斯滤波器");

	f_screem<float>* filter_x = set_f<float>(sf_mode::sobel_x_N);
	Mat soble_x = space_filter_gpu<float, float>("", GSmat, filter_x->len, filter_x->postion, filter_x->data, 1);
	image_show(soble_x, 1, "soble_x");

	f_screem<float>* filter_y = set_f<float>(sf_mode::sobel_y_N);
	Mat soble_y = space_filter_gpu<float, float>("", GSmat, filter_y->len, filter_y->postion, filter_y->data, 1);
	image_show(soble_y, 1, "soble_y");


	Mat M_xy;
	sqrt(soble_x.mul(soble_x) + soble_y.mul(soble_y), M_xy);
	///M_xy.convertTo(M_xy, CV_8U);

	double max1, min1;
	cv::Point min_loc1, max_loc1;
	cv::minMaxLoc(M_xy, &min1, &max1, &min_loc1, &max_loc1);
	cout <<"max1:"<< max1<< endl;
	//计算梯度幅度

	Mat output;
	soble_x.convertTo(soble_x, CV_16SC1);
	soble_y.convertTo(soble_y, CV_16SC1);
	//Canny(Lena,output,max1*0.05,max1*0.15,3,true);
	cv::Canny(soble_x, soble_y, output, max1*0.05, max1*0.15, true);
	image_show(output,1,"canny");

	//计算霍夫曼参数空间
	Mat H=Hough_gpu(output);
	Mat H_N = H.clone();;
	H_N.convertTo(H_N, CV_8U);
	H_N = H_N * 1.5;
	image_show(H_N,0.3,4, "H_N");

	double max, min;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);

	cout << max << endl;
	cout << "x" << max_loc.x << endl;
	cout << "y" << max_loc.y << endl;


	//确定最大直线上的点
	int M = output.rows;
	int N = output.cols;

	output.convertTo(output,CV_8U);

	Mat result=Mat::zeros(output.size(),CV_8U);

	int angle_min = -90;//横向
	int angle_max = 90;

	int P_min = (int)-sqrt(pow(output.rows, 2.0) + pow(output.cols, 2.0));//纵向
	int P_max = (-1.0)*P_min;

	int p_len = P_max + 1;
	int a_len = angle_max + 1;
	
	int x_bnk= max_loc.x;
	cout << "角度：" << max_loc.x - a_len + 1 << endl;
	while (true) {//找到90+的最大元素-1
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//只处理非背景点
				//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//只处理非背景点
				{

					result.at <uchar>(i, j) = 255;
				}

			}
		}
		
		H.at<int>(max_loc.y, max_loc.x) = 0;
		cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);
		//cout << max << endl;
		cout << "x" << max_loc.x << endl;
		cout << "y" << max_loc.y << endl;
		if (x_bnk != max_loc.x || max==0)
			break;
		else
		   x_bnk = max_loc.x;
	}


	////找到-90+1
	int M_H = H.rows;
	//cout<<"x_bnk:"<< -(x_bnk - a_len + 1)+a_len-1 <<endl;
	x_bnk = -(x_bnk - a_len + 1) + a_len - 1;
	cout<<"x_bnk==:"<< x_bnk <<endl;
	int max_big = 0;
	int max_index = 0;
	for (int i = 0; i < M_H ; i++)
	{
		if (H.at<int>(i, x_bnk) > max_big)
		{
			max_big = H.at<int>(i, x_bnk);
			max_index = i;
		}
	}

	max_loc.x = x_bnk;
	max_loc.y = max_index;
	cout<<"角度："<< max_loc.x - a_len + 1 <<endl;
	while (true) {//找到90+的最大元素
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//只处理非背景点
				//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//只处理非背景点
				{

					result.at <uchar>(i, j) = 255;
				}

			}
		}

		H.at<int>(max_loc.y, max_loc.x) = 0;
		cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);
		cout << max << endl;
		cout << "x" << max_loc.x << endl;
		cout << "y" << max_loc.y << endl;
		if (x_bnk != max_loc.x || max == 0)
			break;
	}

	////90du
	//x_bnk = 180;

	//max_big = 0;
	//max_index = 0;
	//for (int i = 0; i < M_H; i++)
	//{
	//	if (H.at<int>(i, x_bnk) > max_big)
	//	{
	//		max_big = H.at<int>(i, x_bnk);
	//		max_index = i;
	//	}
	//}

	//max_loc.x = x_bnk;
	//max_loc.y = max_index;
	//cout << "角度：" << max_loc.x - a_len + 1 << endl;
	//while (true) {//找到90+的最大元素
	//	for (int i = 0; i < M; i++)
	//	{
	//		for (int j = 0; j < N; j++)
	//		{
	//			if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//只处理非背景点
	//			//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//只处理非背景点
	//			{

	//				result.at <uchar>(i, j) = 255;
	//			}

	//		}
	//	}


	//	H.at<int>(max_loc.y, max_loc.x) = 0;
	//	cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);
	//	cout << max << endl;
	//	cout << "x" << max_loc.x << endl;
	//	cout << "y" << max_loc.y << endl;
	//	if (x_bnk != max_loc.x || max == 0)
	//		break;
	//}

	//////90du
	//x_bnk = 0;

	//max_big = 0;
	//max_index = 0;
	//for (int i = 0; i < M_H; i++)
	//{
	//	if (H.at<int>(i, x_bnk) > max_big)
	//	{
	//		max_big = H.at<int>(i, x_bnk);
	//		max_index = i;
	//	}
	//}

	//max_loc.x = x_bnk;
	//max_loc.y = max_index;
	//cout << "角度：" << max_loc.x - a_len + 1 << endl;
	//while (true) {//找到90+的最大元素
	//	for (int i = 0; i < M; i++)
	//	{
	//		for (int j = 0; j < N; j++)
	//		{
	//			if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//只处理非背景点
	//			//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//只处理非背景点
	//			{

	//				result.at <uchar>(i, j) = 255;
	//			}

	//		}
	//	}

	//	H.at<int>(max_loc.y, max_loc.x) = 0;
	//	cv::minMaxLoc(H, &min, &max, &min_loc, &max_loc);
	//	cout << max << endl;
	//	cout << "x" << max_loc.x << endl;
	//	cout << "y" << max_loc.y << endl;
	//	if (x_bnk != max_loc.x || max == 0)
	//		break;
	//}

	//连接result中不超过100的像素缝隙
	image_show(result, 1, "Hough_no_fill2");
	result = Fill_Vertical(result, 150);
	image_show(result, 1, "Hough");
}

void chapter10()
{
	Hough_test();
}