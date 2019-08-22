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
	Mat lena = imread("c:/users/administrator/desktop/opencv/Fig1045a.tif");//无效果图
	cvtColor(lena, lena, COLOR_BGR2GRAY);//转换为灰度图
	Mat lena_o = lena.clone();
	image_show(lena, 1, "原图");
	show_His(lena, "乘积前-直方图", 0);

	Mat pro;
    Mat image_two=Otsu_two(lena, pro);
    image_show(image_two, 1, "2阕值otsu图");
	//7、可变阕值处理
   
}