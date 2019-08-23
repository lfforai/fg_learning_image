#include "opencv_chapter10.cuh"
//һ������任
//�����init=0����Hough_mat,init=1
Mat Hough_gpu(Mat& image_N)//Hough_mat����������
{  
	Mat image = image_N.clone();
	image.convertTo(image, CV_32F);

	int angle_min = -90;//����
	int angle_max = 90;

	int P_min = (int)-sqrt(pow(image.rows, 2.0) + pow(image.cols, 2.0));//����
	int P_max = (-1.0)*P_min;

	int p_len = P_max+1;
	int a_len = angle_max+1;

	Mat Hough_mat = Mat::zeros(2 * P_max + 1, 2 * angle_max + 1, CV_32S);


	int M = image.rows;
	int N = image.cols;
		
	for (int i = 0; i <M; i++)
		{  for (int j = 0; j <N; j++)
			{     
		    if(image.at<float>(i, j) > 0)//ֻ����Ǳ�����
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
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	image_show(Lena, 1, "ԭͼ");
  
	//һ���ȸ�ͨ�˲�����������˹�任
	f_screem<float>* filter_G = set_f<float>(sf_mode::Gauss25_N);
	Mat GSmat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(GSmat, 0.4, "��˹�˲���");

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
	//�����ݶȷ���

	Mat output;
	soble_x.convertTo(soble_x, CV_16SC1);
	soble_y.convertTo(soble_y, CV_16SC1);
	//Canny(Lena,output,max1*0.05,max1*0.15,3,true);
	cv::Canny(soble_x, soble_y, output, max1*0.05, max1*0.15, true);
	image_show(output,1,"canny");

	//��������������ռ�
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


	//ȷ�����ֱ���ϵĵ�
	int M = output.rows;
	int N = output.cols;

	output.convertTo(output,CV_8U);

	Mat result=Mat::zeros(output.size(),CV_8U);

	int angle_min = -90;//����
	int angle_max = 90;

	int P_min = (int)-sqrt(pow(output.rows, 2.0) + pow(output.cols, 2.0));//����
	int P_max = (-1.0)*P_min;

	int p_len = P_max + 1;
	int a_len = angle_max + 1;
	
	int x_bnk= max_loc.x;
	cout << "�Ƕȣ�" << max_loc.x - a_len + 1 << endl;
	while (true) {//�ҵ�90+�����Ԫ��-1
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//ֻ����Ǳ�����
				//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//ֻ����Ǳ�����
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


	////�ҵ�-90+1
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
	cout<<"�Ƕȣ�"<< max_loc.x - a_len + 1 <<endl;
	while (true) {//�ҵ�90+�����Ԫ��
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//ֻ����Ǳ�����
				//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//ֻ����Ǳ�����
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
	//cout << "�Ƕȣ�" << max_loc.x - a_len + 1 << endl;
	//while (true) {//�ҵ�90+�����Ԫ��
	//	for (int i = 0; i < M; i++)
	//	{
	//		for (int j = 0; j < N; j++)
	//		{
	//			if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//ֻ����Ǳ�����
	//			//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//ֻ����Ǳ�����
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
	//cout << "�Ƕȣ�" << max_loc.x - a_len + 1 << endl;
	//while (true) {//�ҵ�90+�����Ԫ��
	//	for (int i = 0; i < M; i++)
	//	{
	//		for (int j = 0; j < N; j++)
	//		{
	//			if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0)) == max_loc.y - p_len + 1)//ֻ����Ǳ�����
	//			//if (output.at<uchar>(i, j) > 0 && (int)(i*cos((max_loc.x - a_len + 1) *3.1415926 / 180.0) + j * sin((max_loc.x - a_len + 1)*3.1415926 / 180.0))>0)//ֻ����Ǳ�����
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

	//����result�в�����100�����ط�϶
	image_show(result, 1, "Hough_no_fill2");
	result = Fill_Vertical(result, 150);
	image_show(result, 1, "Hough");
}


//������ֵ����
//cv::Mat getImageofHistogram(const cv::Mat &hist, int zoom);//����ֱ��ͼ
//void Histogram(Mat& image);ͳ��ֱ��ͼ
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

	/*//ͼ1
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1036a.tif");
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//image_show(Lena, 1, "ԭͼa");

	//Mat Lenaa = imread("C:/Users/Administrator/Desktop/opencv/Fig1036b.tif");
	//cvtColor(Lenaa, Lenaa, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//image_show(Lenaa, 1, "ԭͼb");


	//Mat Lenab = imread("C:/Users/Administrator/Desktop/opencv/Fig1036c.tif");
	//cvtColor(Lenab, Lenab, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//image_show(Lenab, 1, "ԭͼc");
	//
	//Histogram(Lena);
	//Lena=getImageofHistogram(Lena, 1);
	//image_show(Lena, 1, "aԭͼֱ��ͼ");

	//Histogram(Lenaa);
	//Lenaa = getImageofHistogram(Lenaa, 1);
	//image_show(Lenaa, 1, "bԭͼֱ��ͼ");

	//Histogram(Lenab);
	//Lenab = getImageofHistogram(Lenab, 1);
	//image_show(Lenab, 1, "cԭͼֱ��ͼ");*/

	/*//ͼ2
	//Mat Lenaaa = imread("C:/Users/Administrator/Desktop/opencv/Fig1037aa.tif");
	//cvtColor(Lenaaa, Lenaaa, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	////image_show(Lena, 1, "ԭͼab");

	//Mat Lenaab = imread("C:/Users/Administrator/Desktop/opencv/Fig1037ab.tif");
	//cvtColor(Lenaab, Lenaab, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	////image_show(Lenaab, 1, "ԭͼbb");

	//Lenaaa.convertTo(Lenaaa, CV_32F);
	//Lenaab.convertTo(Lenaab, CV_32F);


	//Mat Lenabc = Lenaab+Lenaaa;
	//image_show(Lenabc, 1, "abcԭͼֱ��ͼ");
	//Lenabc.convertTo(Lenabc,CV_8U);

	//Histogram(Lenaab);
	//Lenaab = getImageofHistogram(Lenaab, 1);
	//image_show(Lenaab, 1, "aabԭͼֱ��ͼ");

	//Histogram(Lenabc);
	//Lenabc = getImageofHistogram(Lenabc, 1);
	//image_show(Lenabc, 1, "cbԭͼֱ��ͼ");*/

	//ָ��
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1038a.tif");
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	image_show(Lena, 1, "ָ��ԭͼ");

	Mat Lena_N=Lena.clone();
	Histogram(Lena);
    Lena=getImageofHistogram(Lena, 1);
    image_show(Lena, 1, "ָ��ֱ��ͼ");
	//����125�ǵ�����������ڵ����Ƚϼ�û��д��ش���
	threshold(Lena_N, Lena_N, 125, 255, 0);
	image_show(Lena_N, 1, "ָ����ֵͼ");

	//Otsu

}

//��Ѱrato������ֵ
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

//���԰��Լ������ĸ��ʺ���
Mat Otsu(Mat& image,Mat& probability_N) {
	
	//�����һ��ֱ��ͼ
	Mat Lena=image.clone();
	Mat Lena_N=Lena.clone();
	Histogram(Lena);//row=255
    
	//�������
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
	//ѡ��һ��Kֵ

	//����ƽ���Ҷ�
	//m1(k)�������ؾ�ֵ
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

	//m2(k)�������ʾ�ֵ
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

    //ȫ������ƽ��ֵ
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

	//P1�ۼƸ���
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

	//ȫ�ַ���
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

	//mk�ۼƸ���
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

	//�ֲ�����
	auto var_B = []( float p1, float p2,float mg,float m1 ,float m2, float mk)->float
	{float result;
	 //result = p1 * pow(m1 - mg, 2.0) + p2 * pow(m2 - mg, 2.0);
	 //result = p1*p2*pow((m1-m2),2.0);
	 result = pow(mg*p1 - mk, 2.0)/ (p1*(1 - p1));
	 return result;
	};

	//�������ľֲ�����
	float varG=var_G(probability);//ȫ�ַ���
	float max_var = 0;
	int K=0;
	for(int  k = 0; k < 256; k++)//������󷽲�
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

	//cout<<"���ֵ"<<max_var <<endl;
	//�����n�����ֵ������ƽ��K
	int K_sum = 0;
	int count = 0;
	for (int k = 0; k < 256; k++)//������󷽲�
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

	cout<<"��ֵ��"<< (int)(K_sum/count) <<endl;
	threshold(Lena_N, Lena_N, (int)(K_sum/count), 255, 0);
	//image_show(Lena_N, 1, "�����ֵͼ");
	return  Lena_N.clone();
}


Mat Otsu_two(Mat& image, Mat& probability_N) {

	//�����һ��ֱ��ͼ
	Mat Lena = image.clone();
	Mat Lena_N = Lena.clone();
	Histogram(Lena);//row=255

	//�������
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
	//ѡ��һ��Kֵ

	//����ƽ���Ҷ�
	//m1(k)�������ؾ�ֵ
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

	//Pk�ۼƸ���
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

	//ȫ������ƽ��ֵ
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

	//�ֲ�����
	auto var_B = [](float p1, float p2, float p3,float mg, float m1, float m2, float m3)->float
	{float result;
	 result = p1 * pow(m1 - mg, 2.0) + p2 * pow(m2 - mg, 2.0)+p3*pow(m3-mg,2.0);
	 return result;
	 };

	//�������ľֲ�����
	float max_var = 0;
	int K1 = 0;
	int K2 = 0;
	float mg = mG_avg(probability);
	for(int k1 = 0;k1 < 256; k1++)
	{
	  for(int k2 = 0; k2 < 256; k2++)//������󷽲�
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

	//cout << "��ֵk1��" << (int)K1 << endl;
	//cout << "��ֵk2��" << (int)K2 << endl;

	int K_sum1 = 0;
	int K_sum2 = 0;
	int count = 0;
	for (int k1 = 0; k1 < 256; k1++)
	{
		for (int k2 = 0; k2 < 256; k2++)//������󷽲�
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

	//cout<<"���ֵ"<<max_var <<endl;
	//�����n�����ֵ������ƽ��K
	cout << "��ֵk1��" << (int)(K_sum1 / count) << endl;
	cout << "��ֵk2��" << (int)(K_sum2 / count) << endl;

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

//----------------------------------��ˮ���㷨Դ�����-----------------------------����opencvԴ��
namespace cv_fg
{    //��ԭͼimg�����ͼ��label����һ��ӳ��
	// A node represents a pixel to label
	struct WSNode
	{
		int next;
		int mask_ofs;//���ͼƫ����
		int img_ofs;//ԭͼƫ����
	};

	// Queue for WSNodes
	struct WSQueue  //��¼ wsNode��vector��������һ��λ�ú����һ��λ�õ�index����
	{
		WSQueue() { first = last = 0; }
		int first, last;
	};

	static int
		allocWSNodes(std::vector<WSNode>& storage)
	{
		int sz = (int)storage.size();
		int newsz = MAX(128, sz * 3 / 2);//����һ���Է��䲻����128��wsnode

		storage.resize(newsz);
		if (sz == 0)
		{
			storage[0].next = 0;//��vectorĩβnextѭ��ָ���0������Ԫ��-����next=0���Ǳ�����������β��
			sz = 1;
		}
		for (int i = sz; i < newsz - 1; i++)//Ϊ�����ӵ�vectorԪ�������nextλ�����ӹ�ϵ
			storage[i].next = i + 1;
		storage[newsz - 1].next = 0;//�ﵽԪ��ĩβ���и�ֵ0
		return sz;
	}


	void watershed(InputArray _src, InputOutputArray _markers)
	{  
		// Labels for pixels //
		const int IN_QUEUE = -2; // Pixel visited����ע����ȷ�����б�ˮ��û����ӵĺ�ѡ��
		const int WSHED = -1; // Pixel belongs to watershed //����ˮ�ӵĵ㡢ͼ��߽��

		// possible bit values = 2^8
		const int NQ = 256; //265������

		Mat src = _src.getMat(), dst = _markers.getMat();
	    //���㷨û�иĶ�imgԭͼ���κ����أ�ֻ�ı���ͼ����������ˮ��
		Size size = src.size();
	   
		// Vector of every created node
		std::vector<WSNode> storage;
		int free_node = 0, node;//��¼storage�����¼���Ԫ��λ�ã�node���͵�һ����Ԫ��λ�ã�free_node������node=free_nodeʱ��պô�storage��popһ����
		
		// Priority queue of queues of nodes
		// from high priority (0) to low priority (255) //ˮ�Ӵӵͺ���0һֱ������û���ߺ���255
		WSQueue q[NQ]; //��¼ÿ�����ض�Ӧ��һ����vector��ʼ�ͽ�����λ��
		
		// Non-empty queue with highest priority
		int active_queue;//��ǰˮλ����������������WSQueue�е�λ��
		int i, j;
		
		// Color differences
		int db, dg, dr;//����ͼ��Ĭ�� Scala=3
		int subs_tab[513];//216+216+1

//�����жϱ�����Ϊ�˰�ֵ������255���ڣ��е����
		// MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
        // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

//Create a new node with offsets mofs and iofs in queue idx
//node ��ʾstorage�ĵ�ǰ�����ݵ����һ��Ԫ��λ��,node��next=0��ʾstorage�����ݵ������������������Ϊ��
//free_node ��ʾ��ǰstorage�ĵ�һ����λ��Ԫ��λ��
//storage�ĵ�һ��Ԫ����Զ��0����������κ�����
//�����µ�Ԫ�غ����һ��Ԫ�ص�nextָ��0����storage[0]��Զ��frist=last=0������ֻҪ����һ��WSQueue[idx].first
//q[active_queue].first == 0��˵��������idx�Ѿ������û�����𽥴�ӣ�����idx
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
//����idx������ֵ������һ��storage��Ԫ�أ���������idx��Ӧ��storage�����ֵ
//queue�Ƚ��ȳ���
//ÿ������idx��Ӧһ��storage�е������飬������frist��lastԪ��λ�ü�¼����WSQueue[idex]��һͷһβ
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

//������Ԫ�ص�����ݶȣ����ڷǻҶ�ͼ����3Ԫɫÿ���е����ֵΪ�����ݶȣ����ڻҶ�ͼ��ʵֻ�ü�������һ��
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
		int* mask = dst.ptr<int>(); //����ֵ
		// Step size to next row in mask image
		int mstep = int(dst.step / sizeof(mask[0]));

		//cout << "istep:" << mstep << endl;
		//cout << "src.step:" << dst.step << endl;
		//cout << "sizeof(img[0]):" << sizeof(mask[0]) << endl;

		for (i = 0; i < 256; i++) //�������ֵ����Сֵ��
			subs_tab[i] = 0;
		for (i = 256; i <= 512; i++)
			subs_tab[i] = i - 256;

		// draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
		for (j = 0; j < size.width; j++)
			mask[j] = mask[j + mstep * (size.height - 1)] = WSHED;//top��bottom�б߽������

		// initial phase: put all the neighbor pixels of each marker to the ordered queue -1
		// determine the initial boundaries of the basins
		//�ò���ȷ�㹹���ʼ������أ�Ϊ���������û������-1Ԫ�صĴ����׼���������ʼ��storage
		//�б��ͼ���ص�:
		//1����ȷ��ֻ��mask�еı��Ϊ0�Ĳ���ȷ������Ϊ��Ե��ֻ���ܳ���������
		//2����ʼ��ʱ���ǹ���һ��idx����storage��������һ��idx����
		//3��û��ʹ��8��ͨ��ʹ����4��ͨ
		for (i = 1; i < size.height - 1; i++)
		{
			img += istep; mask += mstep;//ÿִ��һ��ѭ,��תһ��:�������һ�к͵�һ�У��߽�
			mask[0] = mask[size.width - 1] = WSHED; // boundary pixels //��ÿһ�е������б߽������

			for (j = 1; j < size.width - 1; j++)//ÿһ���н���ѭ��
			{
				int* m = mask + j;//�̶�һ��i����j����ɨ��
				if (m[0] < 0) m[0] = 0;
				if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
				{
					// Find smallest difference to adjacent markers
					const uchar* ptr = img + j * 3;//ԭͼ�ϵĵ�
					int idx = 256, t;//���ڲ�����256������أ�����������ʼ��idx

					//����m��0����RGB�������е�һ����ɫ������ݶȣ�Ȼ����4��������Ѱ�ҵ������ɫ�ݶ����ֵ����С��һ����Ϊ��ǰ�ݶ�����ֵ
					if (m[-1] > 0)//��
						c_diff(ptr, ptr - 3, idx);
					
					if (m[1] > 0)//��
					{
						c_diff(ptr, ptr + 3, t);//����÷����ϵ�rgb����ݶ�
						idx = ws_min(idx, t);//����һ�������rgb����ݶȽ��бȽϣ�ѡ���С�ĸ�ֵ��idx
					}
					if (m[-mstep] > 0)//��
					{
						c_diff(ptr, ptr - istep, t);
						idx = ws_min(idx, t);
					}
					if (m[mstep] > 0)//��
					{
						c_diff(ptr, ptr + istep, t);
						idx = ws_min(idx, t);
					}

					// Add to according queue
					assert(0 <= idx && idx <= 255);
					ws_push(idx, i*mstep + j, i*istep + j * 3);//���ݶ�����ֵ��mask��src�е����������У�queue
					m[0] = IN_QUEUE;//��һ�����Ա���Ϊˮ�ĺ���
				}
			}
		}

		// find the first non-empty queue
		//��С�������� ��Сidx���и����ȼ�
		for (i = 0; i < NQ; i++)
			if (q[i].first)
				break;

		// if there is no markers, exit immediately
		if (i == NQ)//NQ=256����������������
			return;

		active_queue = i;//��ʼ������ĵ�һ��idx���صĶ���
		img = src.ptr();
		mask = dst.ptr<int>();

		// recursively fill the basins
		//�ݹ������غ��𽥴��
		for (;;)
		{
			int mofs, iofs;//��ǰ�����mask�Ϻ�src�ϵ������
			int lab = 0, t;
			int* m;
			const uchar* ptr;

			// Get non-empty queue with highest priority 
			// Exit condition: empty priority queue
			// ���first��Ϊ0����idxԪ�ر�ȫ����ûˮλ����������û��һ�����ߺ��ε�idx
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
			//��⵱ǰ���ص����������ĸ����򣬳�ʼ��lab=0
		    //lab��¼����һ����������ش�С�������Χ��m��0���ĵ�ǰ���ص����������Ϸ����maskֵ��һ��
			//������ǰm[0]����ٽ���һ����û��ʹ��ˮλ��û������һ���������������ϣ���ʱ���Ӧ�ðѸõ������ϴ��
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
			m[0] = lab;//����Χһ����Ϊ0�ĵ�����õ��m��0����ֵ����������û��src�ϵ���㣬
			           //�Ѿ���������һ����������һ�������ˡ�

			if (lab == WSHED)//����õ��Ѿ��������˴�ӣ�ֱ�Ӵ��¿�ʼforѭ��������һ��Ϊ-2��Ǳ�ڵ������û�ж�
				continue;

			//����õ㱻��û�ˣ���ô��Ҫ�Ըõ���Χ��Ǳ������Ϊ0���Ľ���һ�����½�����е��ж�
			// Add adjacent, unlabeled pixels to corresponding queue
			if (m[-1] == 0)
			{
				c_diff(ptr, ptr - 3, t);
				ws_push(t, mofs - 1, iofs - 3);
				active_queue = ws_min(active_queue, t);//�жϵ�ǰԤ�����㣬�Ƿ����ȼ����ߣ����ظ�С��
				//������ظ�С����ǰ�ȼ�����Ϊactive_queue
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

//----------------------------------��ˮ���㷨ʹ��-----------------------------
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

	// Return watershed in the form of an image��ͼ�����ʽ���ط�ˮ�� 
	cv::Mat getWatersheds() {

		cv::Mat tmp;
		//�ڱ任ǰ����ÿ������pת��Ϊ255p+255����conertTo��ʵ�֣� 
		markers.convertTo(tmp, CV_8U, 255, 255);

		return tmp.clone();
	}
};

void chapter10()
{
	//1��Hough_test();

    //2\Thresholding_test();

	//3��Otsu()
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1039a.tif");
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//image_show(Lena, 1, "ԭͼ");
	//Mat Lena_show=Lena.clone();
	//Histogram(Lena_show);
	//Lena_show = getImageofHistogram(Lena_show, 1);
	//image_show(Lena_show, 1, "ֱ��ͼ");
	//Mat pro;
	//Lena=Otsu(Lena,pro);
	//image_show(Lena, 1, "otsuͼ");

	//4������ͼ��ƽ������ȫ����ֵ
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1040a.tif");//��Ч��ͼ
	//Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Fig1041a.tif");//��Ч��ͼ
	//cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//image_show(Lena, 1, "ԭͼ");
	//show_His(Lena,"5*5�˲�ǰ-ֱ��ͼ");
	//Mat pro;
	//Mat Lena_show=Otsu(Lena,pro);
	//image_show(Lena_show, 1, "5*5�˲�ǰ-Otsuͼ");

	//f_screem<float>* filter_G = set_f<float>(sf_mode::avg_5);
	//Mat avg_mat = space_filter_gpu<float, float>("", Lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//image_show(avg_mat, 1, "5*5�˲���ͼ");
	//avg_mat.convertTo(avg_mat, CV_8U);
	//show_His(avg_mat, "5*5�˲���-ֱ��ͼ");
	//pro;
	//Lena_show = Otsu(avg_mat,pro);
	//image_show(Lena_show, 1, "5*5�˲���-Otsuͼ");

	//5�����ñ�Ե�Ľ�ȫ����ֵ����
	//1)�ݶ�Ѱ�ұ߽�
	//Mat lena = imread("c:/users/administrator/desktop/opencv/fig1041a.tif");//��Ч��ͼ
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//Mat lena_o=lena.clone();
	//Mat lena_out = lena.clone();
	//image_show(lena, 1, "ԭͼ");
	//show_His(lena,"�˻�ǰ-ֱ��ͼ",0);
	//lena=sobel_grad(lena,1);
	//lena.convertTo(lena, CV_8U);
	//
	//cout<<"��ȡ��ֵ"<<find_Target(lena, 0.997) <<endl;
 //   threshold(lena, lena, find_Target(lena, 0.997), 1, 0);
	//image_show(lena, 1, "�ݶ�ͼ");

	//lena_o=lena.mul(lena_o);
	//image_show(lena_o, 1, "�˻���ͼ");
	//show_His(lena_o, "�˻���-ֱ��ͼ", 1);

	//Histogram(lena_o);
	//lena_o.at<float>(0, 0) = 0;
	//lena_out =Otsu(lena_out, lena_o);
	//image_show(lena_out, 1, "���ͼ");

	//
	//2)��������˹Ѱ�ұ߽� 	Laplace8_N = 6,
	//Mat lena = imread("c:/users/administrator/desktop/opencv/fig1043a.tif");//��Ч��ͼ
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//Mat lena_o = lena.clone();
	//image_show(lena, 1, "ԭͼ");
	//show_His(lena, "�˻�ǰ-ֱ��ͼ", 0);

	//Mat pro;
	//Mat Lena_show=Otsu(lena,pro);
	//image_show(Lena_show, 1, "ֱ��otsuԭͼ");

	//////������˹
	//f_screem<float>* filter_G = set_f<float>(sf_mode::Laplace8_N);
	//Mat laplace_mat = space_filter_gpu<float, float>("", lena, filter_G->len, filter_G->postion, filter_G->data, 1);
	//laplace_mat = abs(laplace_mat);
	//laplace_mat.convertTo(laplace_mat, CV_8U);
	//image_show(laplace_mat, 1, "������˹");
	//
	////99.7%���ݶ�ֵ
	//cout<<"������˹�ݶ���ֵ:"<<find_Target(laplace_mat,0.995)<<endl;

	//threshold(laplace_mat, laplace_mat, find_Target(laplace_mat, 0.995),1, 0);
	//image_show(laplace_mat, 1, "��ֵ�����Ժ��������˹");

	//lena =lena.mul(laplace_mat);
	//image_show(lena, 1, "�˻���ͼ");
	//show_His(lena, "�˻���-ֱ��ͼ", 1);
	//
	//Histogram(lena);
	//lena.at<float>(0, 0) = 0;
	//lena_o =Otsu(lena_o, lena);
	//image_show(lena_o, 1, "���ͼ");

	//6������ֵ����
	//Mat lena = imread("c:/users/administrator/desktop/opencv/Fig1045a.tif");//��Ч��ͼ
	//cvtColor(lena, lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	//Mat lena_o = lena.clone();
	//image_show(lena, 1, "ԭͼ");
	//show_His(lena, "�˻�ǰ-ֱ��ͼ", 0);

	//Mat pro;
 //   Mat image_two=Otsu_two(lena, pro);
 //   image_show(image_two, 1, "2��ֵotsuͼ");
	
 
    //7����ˮ���㷨��һ���ǳ�ţ�Ƶ��㷨
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

	//cv::threshold(binary, binary, 30, 255, THRESH_BINARY_INV);//��ֵ�ָ�ԭͼ�ĻҶ�ͼ����ö�ֵͼ�� 
	// Display the binary image 
	cv::namedWindow("binary Image1");
	cv::imshow("binary Image1", binary);

	// CLOSE operation 
	cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));//5*5�����Σ�8λuchar�ͣ�ȫ1�ṹԪ�� 
	cv::Mat fg1;
	cv::morphologyEx(binary, fg1, cv::MORPH_CLOSE, element5, Point(-1, -1), 1);// ���������������ϸС�ն��������ڽ����� 

	// Display the foreground image 
	cv::namedWindow("Foreground Image");
	cv::imshow("Foreground Image", fg1);

	cv::Mat bg1;
	cv::dilate(binary, bg1, cv::Mat(), cv::Point(-1, -1), 4);//����4�Σ�ê��Ϊ�ṹԪ�����ĵ� 
	cv::threshold(bg1, bg1, 1, 128, cv::THRESH_BINARY_INV);//>=1����������Ϊ128���������� 
	// Display the background image 
	cv::namedWindow("Background Image");
	cv::imshow("Background Image", bg1);
	
	Mat markers1 = fg1 + bg1; //ʹ��Mat������������+���ϲ�ͼ�� 
	cv::namedWindow("markers Image");
	cv::imshow("markers Image", markers1);
	
	// Apply watershed segmentation 
	WatershedSegmenter segmenter1; //ʵ����һ����ˮ��ָ���Ķ��� 
	segmenter1.setMarkers(markers1);//�����㷨�ı��ͼ��ʹ��ˮ�͹��̴�����Ԥ�ȶ���õı�����ؿ�ʼ 
	segmenter1.process(image1);   //������ָ�ԭͼ 

	//Display segmentation result 
	cv::namedWindow("Segmentation1");
	Mat seg=segmenter1.getSegmentation();
	cv::imshow("Segmentation1", segmenter1.getSegmentation());//���޸ĺ�ı��ͼmarkersת��Ϊ����ʾ��8λ�Ҷ�ͼ�����طָ�������ɫΪǰ������ɫΪ������0Ϊ��Ե�� 

	// Display watersheds 
	cv::namedWindow("Watersheds1");
	cv::imshow("Watersheds1", segmenter1.getWatersheds());//��ͼ�����ʽ���ط�ˮ�루�ָ������� 
	//waitKey();
}