#include "opencv_chapter10.cuh"
//一、霍夫变换
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
/*
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
	//} */ 

	//连接result中不超过100的像素缝隙
	image_show(result, 1, "Hough_no_fill2");
	result = Fill_Vertical(result, 150);
	image_show(result, 1, "Hough");
}


//二、阕值分析
//cv::Mat getImageofHistogram(const cv::Mat &hist, int zoom);//绘制直方图
//void Histogram(Mat& image);统计直方图
void show_His(Mat& His_N,char* name,int mode=0) {
	Mat His=His_N.clone();
	His.convertTo(His, CV_8U);
	Histogram(His);
	
	if (mode == 1)
	{
		His.at<float>(0, 0) = 0.0;
	}
  
	His = getImageofHistogram(His, 1);
    image_show(His, 1, name);
}

void Thresholding_test() {

	/*//图1
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1036a.tif");
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	//image_show(Lena, 1, "原图a");

	//Mat Lenaa = imread("C:/Users/Administrator/Desktop/opencv/Fig1036b.tif");
	//cvtColor(Lenaa, Lenaa, COLOR_BGR2GRAY);//转换为灰度图
	//image_show(Lenaa, 1, "原图b");


	//Mat Lenab = imread("C:/Users/Administrator/Desktop/opencv/Fig1036c.tif");
	//cvtColor(Lenab, Lenab, COLOR_BGR2GRAY);//转换为灰度图
	//image_show(Lenab, 1, "原图c");
	//
	//Histogram(Lena);
	//Lena=getImageofHistogram(Lena, 1);
	//image_show(Lena, 1, "a原图直方图");

	//Histogram(Lenaa);
	//Lenaa = getImageofHistogram(Lenaa, 1);
	//image_show(Lenaa, 1, "b原图直方图");

	//Histogram(Lenab);
	//Lenab = getImageofHistogram(Lenab, 1);
	//image_show(Lenab, 1, "c原图直方图");*/

	/*//图2
	//Mat Lenaaa = imread("C:/Users/Administrator/Desktop/opencv/Fig1037aa.tif");
	//cvtColor(Lenaaa, Lenaaa, COLOR_BGR2GRAY);//转换为灰度图
	////image_show(Lena, 1, "原图ab");

	//Mat Lenaab = imread("C:/Users/Administrator/Desktop/opencv/Fig1037ab.tif");
	//cvtColor(Lenaab, Lenaab, COLOR_BGR2GRAY);//转换为灰度图
	////image_show(Lenaab, 1, "原图bb");

	//Lenaaa.convertTo(Lenaaa, CV_32F);
	//Lenaab.convertTo(Lenaab, CV_32F);


	//Mat Lenabc = Lenaab+Lenaaa;
	//image_show(Lenabc, 1, "abc原图直方图");
	//Lenabc.convertTo(Lenabc,CV_8U);

	//Histogram(Lenaab);
	//Lenaab = getImageofHistogram(Lenaab, 1);
	//image_show(Lenaab, 1, "aab原图直方图");

	//Histogram(Lenabc);
	//Lenabc = getImageofHistogram(Lenabc, 1);
	//image_show(Lenabc, 1, "cb原图直方图");*/

	//指纹
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1038a.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	image_show(Lena, 1, "指纹原图");

	Mat Lena_N=Lena.clone();
	Histogram(Lena);
    Lena=getImageofHistogram(Lena, 1);
    image_show(Lena, 1, "指纹直方图");
	//其中125是迭代结果，由于迭代比较简单没有写相关代码
	threshold(Lena_N, Lena_N, 125, 255, 0);
	image_show(Lena_N, 1, "指纹阕值图");

	//Otsu

}

//找寻rato处的阕值
int find_Target(Mat& image_N,float rato){
	Mat image = image_N.clone();
	Histogram(image);
	Scalar ss;
	ss=sum(image);

	for(size_t i = 0; i < 256; i++)
	  {
		image.at<float>(i,0)=image.at<float>(i, 0)/ss[0];
	  }

	int result;
	float sum = 0;
	for (size_t i = 0; i < 256; i++)
	{
		if (sum < rato)
		{
			sum = sum + image.at<float>(i, 0);
		}
		else
		{
			result = i;
			break;
		}
	}
	return result;
}

//可以按自己构建的概率函数
Mat Otsu(Mat& image,Mat& probability_N) {
	
	//计算归一化直方图
	Mat Lena=image.clone();
	Mat Lena_N=Lena.clone();
	Histogram(Lena);//row=255
    
	//计算概率
	Mat probability;
	if (probability_N.empty()){
		Lena.convertTo(probability, CV_32F);
		Scalar ss;
		ss = sum(probability);
		probability = probability / ss[0];
		//cout<< sum(probability)[0] <<endl;
	}
	else {
		probability = probability_N.clone();
		probability.convertTo(probability, CV_32F);
		Scalar ss;
		ss = sum(probability);
		probability = probability / ss[0];
		//cout << sum(probability)[0] << endl;
	}
	//选择一个K值

	//计算平均灰度
	//m1(k)条件像素均值
	auto m1_avg = [](Mat hist_pro,int k)->float
	   {Mat hist=hist_pro.clone();
	    hist.convertTo(hist, CV_32F);

		float sum=0;
		float result=0;
		for(int i = 0; i < k; i++)
		   {   result = result + (float)(i)*hist.at<float>(i, 0);
			   sum = hist.at<float>(i, 0) + sum;
		   }
		result = result / sum;
		return result;
	   };

	//m2(k)条件概率均值
	auto m2_avg = [](Mat hist_pro, int k)->float
	  { int M = 256;
		Mat hist = hist_pro.clone();
		hist.convertTo(hist, CV_32F);

		float sum = 0;
		float result = 0;
		for (int i = k; i < M; i++)
		{
			result = result + (float)(i)*hist.at<float>(i, 0);
			sum = hist.at<float>(i, 0) + sum;
		}
		result = result / sum;
		return result;
	  };

    //全局像素平均值
	auto mG_avg = [](Mat hist_pro)->float
	{int M = 256;
	 Mat hist = hist_pro.clone();
	 hist.convertTo(hist, CV_32F);

	 float result = 0;
	 for (int i = 0; i < M; i++)
	   {
		result = result + (float)(i)*hist.at<float>(i, 0);
	   }
	 return result;
	};

	//P1累计概率
	auto p_sum = [](Mat hist_pro,int k)->float
	{int M = 256;
	 Mat hist = hist_pro.clone();
	 hist.convertTo(hist, CV_32F);

	 float result = 0;
	 for (int i = 0; i < k; i++)
	  {
		result = result + hist.at<float>(i, 0);
	  }
	 return result;
	};

	//全局方差
	auto var_G = [](Mat hist_pro)->float
	{int M = 256;
	 Mat hist = hist_pro.clone();
	 hist.convertTo(hist, CV_32F);

	 float m = 0;
	 for(int i = 0; i < M; i++)
	    {
		 m = m + float(i)*hist.at<float>(i, 0);
	     }

	 float result = 0;
	 for(int i = 0; i < M; i++)
	   {
		 result = result + pow((float)i-m,2.0)*hist.at<float>(i, 0);
	    }
	  return result;
	};

	//mk累计概率
	auto mk_sum = [](Mat hist_pro, int k)->float
	{int M = 256;
	Mat hist = hist_pro.clone();
	hist.convertTo(hist, CV_32F);

	float result = 0;
	for (int i = 0; i < k; i++)
	{
		result = result + (float)(i)*hist.at<float>(i, 0);
	}
	return result;
	};

	//局部方差
	auto var_B = []( float p1, float p2,float mg,float m1 ,float m2, float mk)->float
	{float result;
	 //result = p1 * pow(m1 - mg, 2.0) + p2 * pow(m2 - mg, 2.0);
	 //result = p1*p2*pow((m1-m2),2.0);
	 result = pow(mg*p1 - mk, 2.0)/ (p1*(1 - p1));
	 return result;
	};

	//计算最大的局部方差
	float varG=var_G(probability);//全局方差
	float max_var = 0;
	int K=0;
	for(int  k = 0; k < 256; k++)//计算最大方差
	{
		float p1 = p_sum(probability, k);
		float p2 = 1 - p1;

		float mG = mG_avg(probability);
		float m1 = m1_avg(probability, k);
		float m2 = m2_avg(probability, k);

		float mk=mk_sum(probability, k);

		float v_B=var_B(p1,p2,mG,m1,m2,mk);
		if (v_B >= max_var) {
			max_var = v_B;
			K = k;
		}
		//cout<<v_B <<"|"<<k<<endl;
	 }

	//cout<<"最大值"<<max_var <<endl;
	//如果有n个最大值，计算平均K
	int K_sum = 0;
	int count = 0;
	for (int k = 0; k < 256; k++)//计算最大方差
	{
		float p1 = p_sum(probability, k);
		float p2 = 1 - p1;

		float mG = mG_avg(probability);
		float m1 = m1_avg(probability, k);
		float m2 = m2_avg(probability, k);

		float mk = mk_sum(probability, k);

		float v_B = var_B(p1, p2, mG, m1, m2, mk);
		if (v_B == max_var) {
			K_sum+=k;
			count+=1;
		}
	}

	cout<<"阕值："<< (int)(K_sum/count) <<endl;
	threshold(Lena_N, Lena_N, (int)(K_sum/count), 255, 0);
	//image_show(Lena_N, 1, "最大阕值图");
	return  Lena_N.clone();
}


Mat Otsu_two(Mat& image, Mat& probability_N) {

	//计算归一化直方图
	Mat Lena = image.clone();
	Mat Lena_N = Lena.clone();
	Histogram(Lena);//row=255

	//计算概率
	Mat probability;
	if (probability_N.empty()) {
		Lena.convertTo(probability, CV_32F);
		Scalar ss;
		ss = sum(probability);
		probability = probability / ss[0];
		//cout<<"lena:" <<sum(probability)[0] <<endl;
	}
	else {
		probability = probability_N.clone();
		probability.convertTo(probability, CV_32F);
		Scalar ss;
		ss = sum(probability);
		probability = probability / ss[0];
		//cout <<"probability_N"<<sum(probability)[0] << endl;
	}
	//选择一个K值

	//计算平均灰度
	//m1(k)条件像素均值
	auto mk_avg = [](Mat hist_pro, int k1,int k2)->float
	{Mat hist = hist_pro.clone();
	hist.convertTo(hist, CV_32F);

	float sum = 0;
	float result = 0;
	for (int i = k1; i < k2; i++)
	{
		result = result + (float)(i)*hist.at<float>(i, 0);
		sum = hist.at<float>(i, 0) + sum;
	}
	result = result / sum;
	return result;
	};

	//Pk累计概率
	auto p_sum = [](Mat hist_pro, int k1,int k2)->float
	{int M = 256;
	Mat hist = hist_pro.clone();
	hist.convertTo(hist, CV_32F);

	float result = 0;
	for (int i = k1; i < k2; i++)
	{
		result = result + hist.at<float>(i, 0);
	}
	return result;
	};

	//全局像素平均值
	auto mG_avg = [](Mat hist_pro)->float
	{int M = 256;
	 Mat hist = hist_pro.clone();
	 hist.convertTo(hist, CV_32F);

	 float result = 0;
	 for (int i = 0; i < M; i++)
	  {
		result = result + (float)(i)*hist.at<float>(i, 0);
	  }
	  return result;
	 };

	//局部方差
	auto var_B = [](float p1, float p2, float p3,float mg, float m1, float m2, float m3)->float
	{float result;
	 result = p1 * pow(m1 - mg, 2.0) + p2 * pow(m2 - mg, 2.0)+p3*pow(m3-mg,2.0);
	 return result;
	 };

	//计算最大的局部方差
	float max_var = 0;
	int K1 = 0;
	int K2 = 0;
	float mg = mG_avg(probability);
	for(int k1 = 0;k1 < 256; k1++)
	{
	  for(int k2 = 0; k2 < 256; k2++)//计算最大方差
		{      
			if (k1<k2) 
			{   //023-63855237
				float p1=p_sum(probability, 0, k1);
				float p2=p_sum(probability, k1, k2);
				float p3=p_sum(probability, k2, 256);
				float m1 = mk_avg(probability,0,k1);
				float m2 = mk_avg(probability, k1, k2);
				float m3 = mk_avg(probability, k2, 256);
			
				float v_B = var_B(p1,p2,p3,mg,m1,m2,m3);
				if(v_B > max_var)
				  {
					max_var = v_B;
					K1 = k1;
					K2 = k2;
					//cout <<p1+p2+p3<<"|"<<v_B << "|" << k1 << "|" << k2 << endl;
				  }
			}
			
		}
	}

	//cout << "阕值k1：" << (int)K1 << endl;
	//cout << "阕值k2：" << (int)K2 << endl;

	int K_sum1 = 0;
	int K_sum2 = 0;
	int count = 0;
	for (int k1 = 0; k1 < 256; k1++)
	{
		for (int k2 = 0; k2 < 256; k2++)//计算最大方差
		{
			if (k1 < k2)
			{
				float p1 = p_sum(probability, 0, k1);
				float p2 = p_sum(probability, k1, k2);
				float p3 = p_sum(probability, k2, 256);
				float m1 = mk_avg(probability, 0, k1);
				float m2 = mk_avg(probability, k1, k2);
				float m3 = mk_avg(probability, k2, 256);

				float v_B = var_B(p1, p2, p3, mg, m1, m2, m3);
				if (v_B == max_var)
				{
					K_sum1 += k1;
					K_sum2 += k2;
					count += 1;
				}
			}
			//cout<<v_B <<"|"<<k<<endl;
		}
	}

	//cout<<"最大值"<<max_var <<endl;
	//如果有n个最大值，计算平均K
	cout << "阕值k1：" << (int)(K_sum1 / count) << endl;
	cout << "阕值k2：" << (int)(K_sum2 / count) << endl;

	K1 = (int)(K_sum1 / count);
	K2 = (int)(K_sum2 / count);

	int N = Lena_N.cols;
	int M = Lena_N.rows;
	for (size_t i = 0; i <M; i++)
		  {  for (size_t j = 0; j <N; j++)
			   { 
			    if(Lena_N.at<uchar>(i, j)<=K1)
			       Lena_N.at<uchar>(i, j)=0;
				if(Lena_N.at<uchar>(i, j) >K1 && Lena_N.at<uchar>(i, j)<=K2)
				   Lena_N.at<uchar>(i, j) = 155;
				if(Lena_N.at<uchar>(i, j) >K2)
				   Lena_N.at<uchar>(i, j) = 255;
			   }
		  }
	return  Lena_N.clone();
}

//----------------------------------分水岭算法源码解析-----------------------------来自opencv源码
namespace cv_fg
{    //从原图img到标记图（label）的一个映射
	// A node represents a pixel to label
	struct WSNode
	{
		int next;
		int mask_ofs;//标记图偏移量
		int img_ofs;//原图偏移量
	};

	// Queue for WSNodes
	struct WSQueue  //记录 wsNode的vector子向量第一个位置和最后一个位置的index索引
	{
		WSQueue() { first = last = 0; }
		int first, last;
	};

	static int
		allocWSNodes(std::vector<WSNode>& storage)
	{
		int sz = (int)storage.size();
		int newsz = MAX(128, sz * 3 / 2);//最少一次性分配不少于128个wsnode

		storage.resize(newsz);
		if (sz == 0)
		{
			storage[0].next = 0;//在vector末尾next循环指向第0个向量元素-遇到next=0就是表明到达向量尾部
			sz = 1;
		}
		for (int i = sz; i < newsz - 1; i++)//为新增加的vector元素链添加next位置链接关系
			storage[i].next = i + 1;
		storage[newsz - 1].next = 0;//达到元素末尾进行赋值0
		return sz;
	}


	void watershed(InputArray _src, InputOutputArray _markers)
	{  
		// Labels for pixels //
		const int IN_QUEUE = -2; // Pixel visited，标注不明确领域中被水淹没或逐坝的候选点
		const int WSHED = -1; // Pixel belongs to watershed //属于水坝的点、图像边界点

		// possible bit values = 2^8
		const int NQ = 256; //265个像素

		Mat src = _src.getMat(), dst = _markers.getMat();
	    //本算法没有改动img原图的任何像素，只改变标记图像用来建筑水坝
		Size size = src.size();
	   
		// Vector of every created node
		std::vector<WSNode> storage;
		int free_node = 0, node;//记录storage中最新加入元素位置（node）和第一个空元素位置（free_node），当node=free_node时候刚好从storage中pop一个点
		
		// Priority queue of queues of nodes
		// from high priority (0) to low priority (255) //水坝从低海拔0一直向上淹没到高海拔255
		WSQueue q[NQ]; //记录每个像素对应的一个子vector开始和结束的位置
		
		// Non-empty queue with highest priority
		int active_queue;//当前水位漫过点所在像素在WSQueue中的位置
		int i, j;
		
		// Color differences
		int db, dg, dr;//输入图像默认 Scala=3
		int subs_tab[513];//216+216+1

//不用判断表达？，是为了把值控制在255以内，有点多余
		// MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
        // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

//Create a new node with offsets mofs and iofs in queue idx
//node 表示storage的当前有数据的最后一个元素位置,node的next=0表示storage有数据的数组结束，后面数组为空
//free_node 表示当前storage的第一个空位置元素位置
//storage的第一个元素永远是0，不会放入任何数据
//放入新的元素后，最后一个元素的next指向0，而storage[0]永远是frist=last=0，所以只要发现一个WSQueue[idx].first
//q[active_queue].first == 0就说明该像素idx已经完成淹没或者逐渐大坝，可以idx
#define ws_push(idx,mofs,iofs)              \
    {                                       \
        if( !free_node )                    \
            free_node = allocWSNodes( storage );\
        node = free_node;                   \
        free_node = storage[free_node].next;\
        storage[node].next = 0;             \
        storage[node].mask_ofs = mofs;      \
        storage[node].img_ofs = iofs;       \
        if( q[idx].last )                   \
            storage[q[idx].last].next=node; \
        else                                \
            q[idx].first = node;            \
        q[idx].last = node;                 \
    }

//Get next node from queue idx
//根据idx的像素值，弹出一个storage的元素，并调整该idx对应子storage数组的值
//queue先进先出的
//每个像素idx对应一个storage中的子数组，子数组frist和last元素位置记录了在WSQueue[idex]中一头一尾
//
#define ws_pop(idx,mofs,iofs)               \
    {                                       \
        node = q[idx].first;                \
        q[idx].first = storage[node].next;  \
        if( !storage[node].next )           \
            q[idx].last = 0;                \
        storage[node].next = free_node;     \
        free_node = node;                   \
        mofs = storage[node].mask_ofs;      \
        iofs = storage[node].img_ofs;       \
    }

//求相邻元素的最大梯度，对于非灰度图计算3元色每个中的最大值为代表梯度，对于灰度图其实只用计算其中一个
// Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
    {                                        \
        db = std::abs((ptr1)[0] - (ptr2)[0]);\
        dg = std::abs((ptr1)[1] - (ptr2)[1]);\
        dr = std::abs((ptr1)[2] - (ptr2)[2]);\
        diff = ws_max(db,dg);                \
        diff = ws_max(diff,dr);              \
        assert( 0 <= diff && diff <= 255 );  \
    }

		CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_32SC1);
		CV_Assert(src.size() == dst.size());

		// Current pixel in input image
		const uchar* img = src.ptr();
		// Step size to next row in input image
		int istep = int(src.step / sizeof(img[0]));
		//cout<<"istep:"<<istep<<endl;
		//cout << "src.step:" << src.step << endl;
		//cout << "sizeof(img[0]):" << sizeof(img[0]) << endl;

		// Current pixel in mask image
		int* mask = dst.ptr<int>(); //返回值
		// Step size to next row in mask image
		int mstep = int(dst.step / sizeof(mask[0]));

		//cout << "istep:" << mstep << endl;
		//cout << "src.step:" << dst.step << endl;
		//cout << "sizeof(img[0]):" << sizeof(mask[0]) << endl;

		for (i = 0; i < 256; i++) //计算最大值和最小值用
			subs_tab[i] = 0;
		for (i = 256; i <= 512; i++)
			subs_tab[i] = i - 256;

		// draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
		for (j = 0; j < size.width; j++)
			mask[j] = mask[j + mstep * (size.height - 1)] = WSHED;//top和bottom行边界筑大坝

		// initial phase: put all the neighbor pixels of each marker to the ordered queue -1
		// determine the initial boundaries of the basins
		//用不明确点构造初始化的盆地，为后面盆地淹没和修筑-1元素的大坝做准备，构造初始化storage
		//有标记图的特点:
		//1、明确了只淹mask中的标记为0的不明确区域，因为边缘点只可能出现在这里
		//2、初始化时候是构造一批idx进入storage，而不是一个idx像素
		//3、没有使用8连通，使用了4连通
		for (i = 1; i < size.height - 1; i++)
		{
			img += istep; mask += mstep;//每执行一次循,跳转一行:跳过最后一行和第一行：边界
			mask[0] = mask[size.width - 1] = WSHED; // boundary pixels //把每一行的两端列边界筑大坝

			for (j = 1; j < size.width - 1; j++)//每一列中进行循环
			{
				int* m = mask + j;//固定一行i，按j逐列扫描
				if (m[0] < 0) m[0] = 0;
				if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
				{
					// Find smallest difference to adjacent markers
					const uchar* ptr = img + j * 3;//原图上的点
					int idx = 256, t;//由于不存在256这个像素，所以用来初始化idx

					//计算m【0】的RGB三像素中的一个颜色的最大梯度，然后在4个方向上寻找当这个颜色梯度最大值中最小的一个作为当前梯度像素值
					if (m[-1] > 0)//左
						c_diff(ptr, ptr - 3, idx);
					
					if (m[1] > 0)//右
					{
						c_diff(ptr, ptr + 3, t);//计算该方向上的rgb最大梯度
						idx = ws_min(idx, t);//与上一个方向的rgb最大梯度进行比较，选择较小的赋值给idx
					}
					if (m[-mstep] > 0)//上
					{
						c_diff(ptr, ptr - istep, t);
						idx = ws_min(idx, t);
					}
					if (m[mstep] > 0)//下
					{
						c_diff(ptr, ptr + istep, t);
						idx = ws_min(idx, t);
					}

					// Add to according queue
					assert(0 <= idx && idx <= 255);
					ws_push(idx, i*mstep + j, i*istep + j * 3);//把梯度向量值在mask和src中的坐标放入队列，queue
					m[0] = IN_QUEUE;//第一批可以被作为水的函数
				}
			}
		}

		// find the first non-empty queue
		//从小到大搜索 ，小idx具有高优先级
		for (i = 0; i < NQ; i++)
			if (q[i].first)
				break;

		// if there is no markers, exit immediately
		if (i == NQ)//NQ=256，不存在这种像素
			return;

		active_queue = i;//初始化计算的第一个idx像素的队列
		img = src.ptr();
		mask = dst.ptr<int>();

		// recursively fill the basins
		//递归计算盆地和逐渐大坝
		for (;;)
		{
			int mofs, iofs;//当前计算的mask上和src上的坐标点
			int lab = 0, t;
			int* m;
			const uchar* ptr;

			// Get non-empty queue with highest priority 
			// Exit condition: empty priority queue
			// 如果first变为0，该idx元素被全部淹没水位上升开会淹没下一个更高海拔的idx
			if (q[active_queue].first == 0)
			{
				for (i = active_queue + 1; i < NQ; i++)
					if (q[i].first)
						break;
				if (i == NQ)
					break;
				active_queue = i;
			}

			// Get next node
			ws_pop(active_queue, mofs, iofs);

			// Calculate pointer to current pixel in input and marker image
			m = mask + mofs;
			ptr = img + iofs;

			// Check surrounding pixels for labels
			// to determine label for current pixel
			//检测当前像素的上下左右四个方向，初始化lab=0
		    //lab记录了上一个方向的像素大小，如果在围绕m【0】的当前像素的两个或以上方向的mask值不一致
			//表明当前m[0]如果再进行一次淹没会使得水位淹没到另外一个背景或者物体上，这时候就应该把该点修筑上大坝
			t = m[-1]; // Left
			if (t > 0) lab = t;
			
			t = m[1]; // Right
			if (t > 0)
			{
				if (lab == 0) lab = t;
				else if (t != lab) lab = WSHED;
			}
			
			t = m[-mstep]; // Top
			if (t > 0)
			{
				if (lab == 0) lab = t;
				else if (t != lab) lab = WSHED;
			}
			
			t = m[mstep]; // Bottom
			if (t > 0)
			{
				if (lab == 0) lab = t;
				else if (t != lab) lab = WSHED;
			}

			// Set label to current pixel in marker image
			assert(lab != 0);
			m[0] = lab;//用周围一个不为0的点替代该点的m【0】的值，表明被淹没的src上的这点，
			           //已经被归属于一个背景或者一个物体了。

			if (lab == WSHED)//如果该点已经修筑上了大坝，直接从新开始for循环，对下一个为-2的潜在点进行淹没判断
				continue;

			//如果该点被淹没了，那么需要对该点周围的潜在区域（为0）的进行一次重新进入队列的判断
			// Add adjacent, unlabeled pixels to corresponding queue
			if (m[-1] == 0)
			{
				c_diff(ptr, ptr - 3, t);
				ws_push(t, mofs - 1, iofs - 3);
				active_queue = ws_min(active_queue, t);//判断当前预出来点，是否优先级更高（像素更小）
				//如果像素更小，当前等级升高为active_queue
				m[-1] = IN_QUEUE;
			}
			if (m[1] == 0)
			{
				c_diff(ptr, ptr + 3, t);
				ws_push(t, mofs + 1, iofs + 3);
				active_queue = ws_min(active_queue, t);
				m[1] = IN_QUEUE;
			}
			if (m[-mstep] == 0)
			{
				c_diff(ptr, ptr - istep, t);
				ws_push(t, mofs - mstep, iofs - istep);
				active_queue = ws_min(active_queue, t);
				m[-mstep] = IN_QUEUE;
			}
			if (m[mstep] == 0)
			{
				c_diff(ptr, ptr + istep, t);
				ws_push(t, mofs + mstep, iofs + istep);
				active_queue = ws_min(active_queue, t);
				m[mstep] = IN_QUEUE;
			}
		}
	}
}

//----------------------------------分水岭算法使用-----------------------------
class WatershedSegmenter {

private:

public:
	cv::Mat markers;

	void setMarkers(const cv::Mat& markerImage) {

		// Convert to image of ints 
		markerImage.convertTo(markers, CV_32S);
	}

	cv::Mat process(const cv::Mat &image) {

		// Apply watershed 
		cv_fg::watershed(image, markers);

		return markers.clone();
	}

	// Return result in the form of an image 
	cv::Mat getSegmentation() {

		cv::Mat tmp;
		// all segment with label higher than 255 
		// will be assigned value 255 
		markers.convertTo(tmp, CV_8U);

		return tmp.clone();
	}

	// Return watershed in the form of an image以图像的形式返回分水岭 
	cv::Mat getWatersheds() {

		cv::Mat tmp;
		//在变换前，把每个像素p转换为255p+255（在conertTo中实现） 
		markers.convertTo(tmp, CV_8U, 255, 255);

		return tmp.clone();
	}
};

void chapter10()
{
	//1、Hough_test();

    //2\Thresholding_test();

	//3、Otsu()
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1039a.tif");
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	//image_show(Lena, 1, "原图");
	//Mat Lena_show=Lena.clone();
	//Histogram(Lena_show);
	//Lena_show = getImageofHistogram(Lena_show, 1);
	//image_show(Lena_show, 1, "直方图");
	//Mat pro;
	//Lena=Otsu(Lena,pro);
	//image_show(Lena, 1, "otsu图");

	//4、利用图像平滑改善全局阕值
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1040a.tif");//有效果图
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1041a.tif");//无效果图
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图
	//image_show(Lena, 1, "原图");
	//show_His(Lena,"5*5滤波前-直方图");
	//Mat pro;
	//Mat Lena_show=Otsu(Lena,pro);
	//image_show(Lena_show, 1, "5*5滤波前-Otsu图");

	//f_screem<float>* filter_G = set_f<float>(sf_mode::avg_5);
	//Mat avg_mat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(avg_mat, 1, "5*5滤波后图");
	//avg_mat.convertTo(avg_mat, CV_8U);
	//show_His(avg_mat, "5*5滤波后-直方图");
	//pro;
	//Lena_show = Otsu(avg_mat,pro);
	//image_show(Lena_show, 1, "5*5滤波后-Otsu图");

	//5、利用边缘改进全局阕值处理
	//1)梯度寻找边界
	//Mat lena = imread("c:/users/administrator/desktop/opencv/fig1041a.tif");//无效果图
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//转换为灰度图
	//Mat lena_o=lena.clone();
	//Mat lena_out = lena.clone();
	//image_show(lena, 1, "原图");
	//show_His(lena,"乘积前-直方图",0);
	//lena=sobel_grad(lena,1);
	//lena.convertTo(lena, CV_8U);
	//
	//cout<<"提取阕值"<<find_Target(lena, 0.997) <<endl;
 //   threshold(lena, lena, find_Target(lena, 0.997), 1, 0);
	//image_show(lena, 1, "梯度图");

	//lena_o=lena.mul(lena_o);
	//image_show(lena_o, 1, "乘积后图");
	//show_His(lena_o, "乘积后-直方图", 1);

	//Histogram(lena_o);
	//lena_o.at<float>(0, 0) = 0;
	//lena_out =Otsu(lena_out, lena_o);
	//image_show(lena_out, 1, "结果图");

	//
	//2)用拉普拉斯寻找边界 	Laplace8_N = 6,
	//Mat lena = imread("c:/users/administrator/desktop/opencv/fig1043a.tif");//无效果图
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//转换为灰度图
	//Mat lena_o = lena.clone();
	//image_show(lena, 1, "原图");
	//show_His(lena, "乘积前-直方图", 0);

	//Mat pro;
	//Mat Lena_show=Otsu(lena,pro);
	//image_show(Lena_show, 1, "直接otsu原图");

	//////拉普拉斯
	//f_screem<float>* filter_G = set_f<float>(sf_mode::Laplace8_N);
	//Mat laplace_mat = space_filter_gpu<float, float>("", lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//laplace_mat = abs(laplace_mat);
	//laplace_mat.convertTo(laplace_mat, CV_8U);
	//image_show(laplace_mat, 1, "拉普拉斯");
	//
	////99.7%的梯度值
	//cout<<"拉普莱斯梯度阕值:"<<find_Target(laplace_mat,0.995)<<endl;

	//threshold(laplace_mat, laplace_mat, find_Target(laplace_mat, 0.995),1, 0);
	//image_show(laplace_mat, 1, "阕值处理以后的拉普拉斯");

	//lena =lena.mul(laplace_mat);
	//image_show(lena, 1, "乘积后图");
	//show_His(lena, "乘积后-直方图", 1);
	//
	//Histogram(lena);
	//lena.at<float>(0, 0) = 0;
	//lena_o =Otsu(lena_o, lena);
	//image_show(lena_o, 1, "结果图");

	//6、多阕值处理
	//Mat lena = imread("c:/users/administrator/desktop/opencv/Fig1045a.tif");//无效果图
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//转换为灰度图
	//Mat lena_o = lena.clone();
	//image_show(lena, 1, "原图");
	//show_His(lena, "乘积前-直方图", 0);

	//Mat pro;
 //   Mat image_two=Otsu_two(lena, pro);
 //   image_show(image_two, 1, "2阕值otsu图");
	
 
    //7、分水领算法，一个非常牛逼的算法
	// Read input image 
	cv::Mat image1 = cv::imread("c:/users/administrator/desktop/opencv/Fig1056a.tif");
	// Display the color image 
	cv::resize(image1, image1, cv::Size(), 2, 2);
	cv::namedWindow("Original Image1");
	cv::imshow("Original Image1", image1);

	Mat binary;
	cv::cvtColor(image1, binary, COLOR_BGRA2GRAY);
	Mat binary_O=binary.clone();
	Mat pro;
	binary=Otsu(binary, pro);
	binary.convertTo(binary,CV_8U);

	//cv::threshold(binary, binary, 30, 255, THRESH_BINARY_INV);//阈值分割原图的灰度图，获得二值图像 
	// Display the binary image 
	cv::namedWindow("binary Image1");
	cv::imshow("binary Image1", binary);

	// CLOSE operation 
	cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));//5*5正方形，8位uchar型，全1结构元素 
	cv::Mat fg1;
	cv::morphologyEx(binary, fg1, cv::MORPH_CLOSE, element5, Point(-1, -1), 1);// 闭运算填充物体内细小空洞、连接邻近物体 

	// Display the foreground image 
	cv::namedWindow("Foreground Image");
	cv::imshow("Foreground Image", fg1);

	cv::Mat bg1;
	cv::dilate(binary, bg1, cv::Mat(), cv::Point(-1, -1), 4);//膨胀4次，锚点为结构元素中心点 
	cv::threshold(bg1, bg1, 1, 128, cv::THRESH_BINARY_INV);//>=1的像素设置为128（即背景） 
	// Display the background image 
	cv::namedWindow("Background Image");
	cv::imshow("Background Image", bg1);
	
	Mat markers1 = fg1 + bg1; //使用Mat类的重载运算符+来合并图像。 
	cv::namedWindow("markers Image");
	cv::imshow("markers Image", markers1);
	
	// Apply watershed segmentation 
	WatershedSegmenter segmenter1; //实例化一个分水岭分割方法的对象 
	segmenter1.setMarkers(markers1);//设置算法的标记图像，使得水淹过程从这组预先定义好的标记像素开始 
	segmenter1.process(image1);   //传入待分割原图 

	//Display segmentation result 
	cv::namedWindow("Segmentation1");
	Mat seg=segmenter1.getSegmentation();
	cv::imshow("Segmentation1", segmenter1.getSegmentation());//将修改后的标记图markers转换为可显示的8位灰度图并返回分割结果（白色为前景，灰色为背景，0为边缘） 

	// Display watersheds 
	cv::namedWindow("Watersheds1");
	cv::imshow("Watersheds1", segmenter1.getWatersheds());//以图像的形式返回分水岭（分割线条） 
	//waitKey();
}