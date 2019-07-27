#include "opencv_chapter4.h"

//����4.15
void space2frequency() 
{
	//�ο�΢��https://blog.csdn.net/lvsehaiyang1993/article/details/80876712
		//��4.15��ģ�嵽Ƶ���˲���
	Mat lena = imread("C:/Users/Administrator/Desktop/opencv/house.png", IMREAD_GRAYSCALE);
	Mat z = Mat::zeros(316, 316, CV_32F);
	resize(lena, lena, z.size(), INTER_CUBIC);
	Mat dft_lena = lena.clone();//Ƶ���˲���
	Mat dft_lena_filter_space = lena.clone();//�ռ��˲���ͼ
	imshow("ԭͼhouse:", lena);
	house_test(lena);
	move_to_center(lena);
	imshow("logƵ��house:", lena);

	//һ��Ƶ���˲��Ĵ�С  
	//��չΪP*Q,
	/*copyMakeBorder(dft_lena, dft_lena,1, 1, 1,1,
		BORDER_CONSTANT, Scalar::all(0));*/
	resize_tpye* re = graph_resize(dft_lena);
	dft_lena.convertTo(dft_lena, CV_8U);
	imshow("ԭͼP*Q��չ�Ժ�ͼ:", dft_lena);
	cout << "��Ҫ�����Ĵ�С��" << re->size_cols << "|" << re->size_rows << endl;
	Mat real;
	Mat ima;

	Mat h = Mat::zeros(dft_lena.size(), CV_32F);
	cout << "������ż�������Һ�ԭͼƵ����һ����" << h.size() << endl;//�����Ǹ�ż��
	int r_m = (int)(h.rows / 2.0) - 1;
	int c_n = (int)(h.cols / 2.0) - 1;

	//h.at<float>(1, 0) = 2;  //����������λ�ø���Ҷ�任�Ժ�������溯��
	//h.at<float>(1, 1) = 1;
	//h.at<float>(1, h.cols - 1) = 1;
	//h.at<float>(h.rows - 1, 0) = -2;
	//h.at<float>(h.rows - 1, 1) = -1;
	//h.at<float>(h.rows - 1, h.cols - 1) = -1;

	h.at<float>(r_m, c_n) = -1;  //����������λ�ø���Ҷ�任�Ժ�������溯��
	h.at<float>(r_m, c_n + 1) = -2;
	h.at<float>(r_m, c_n + 2) = -1;
	h.at<float>(r_m + 1, c_n) = 0;
	h.at<float>(r_m + 1, c_n + 1) = 0;
	h.at<float>(r_m + 1, c_n + 2) = 0;
	h.at<float>(r_m + 2, c_n) = 1;
	h.at<float>(r_m + 2, c_n + 1) = 2;
	h.at<float>(r_m + 2, c_n + 2) = 1;

	for (int i = 0; i < h.rows; i++)
	{
		for (int j = 0; j < h.cols; j++)
		{
			h.at<float>(i, j) = h.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	//ż�����鿴
	Met_oe_info*  de_ifo=Mat_is_odd_or_even(h);
	de_ifo->print();
	
	//���˲���ת����Ƶ����
	Mat h_dft = fast_dft(h, real, ima);
	amplitude_common(h_dft);
	h_dft.convertTo(h_dft, CV_8U);
	imshow("�˲���Ƶ��ͼ:", h_dft);
	for (int i = 0; i < ima.rows; i++)
	{
		for (int j = 0; j < ima.cols; j++)
		{
			ima.at<float>(i, j) = ima.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	for (int i = 0; i < real.rows; i++)
	{
		for (int j = 0; j < real.cols; j++)
		{
			real.at<float>(i, j) = 0.0;
		}
	}
	
	//���ע�Ϳ��԰������Ϊʲô��Ҫ���ԣ�-1��^(u+v)*F(U,V)
	Mat real_N;
	Mat ima_N;
	Mat h_tepm[] = {real,ima};
	merge(h_tepm,2,h);
	fourior_inverser(h, real_N, ima_N);
	divide(real_N,real_N.rows*real_N.cols,real_N);
	real_N.convertTo(real_N, CV_32S);
	ima_N.convertTo(ima_N,CV_32S);
	Scalar ss = sum(ima_N);
	cout<< ss[0]<<endl;
	for (size_t i = 0; i < real_N.rows; i++)
	{
	  for (size_t j = 0; j <real_N.cols; j++)
		{
			if (abs(real_N.at<int>(i,j))>0) {
				cout << "row:" << i << "|col:" << j << ",value:="<< real_N.at<int>(i, j) << endl;
			}
		}
	}

	//������ʽ�˲�����\
	//1)������С����0
	//2)����F(U��V)����Ҷ�任
	Mat dft_lena_filter;
	dft_lena.convertTo(dft_lena_filter, CV_32F);
	for (int i = 0; i < dft_lena_filter.rows; i++)
	{
		for (int j = 0; j < dft_lena_filter.cols; j++)
		{
			dft_lena_filter.at<float>(i, j) = dft_lena_filter.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	Mat real_src_filter;
	Mat ima_src_filter;
	Mat src_image_dft = fast_dft(dft_lena_filter, real_src_filter, ima_src_filter);

	//3)���Ļ��ƶ�
	//move_to_center(src_image_dft);

	//4)�����˲���ͼ�񣬵�����С��������p/2��Q/2�ط�

	//5)������
	//src_image_dft = src_image_dft.mul(h_dft);
	Mat vector[] = { Mat::zeros(real.size(),CV_32F),Mat::zeros(ima.size(),CV_32F) };
	split(src_image_dft, vector);
	Mat image_real = vector[0];
	Mat image_ima = vector[1];
	real = Mat::zeros(real.size(), CV_32F);

	for (size_t i = 0; i < image_real.rows; i++)
	{
		for (size_t j = 0; j < image_real.cols; j++)
		{   //real ȫ��Ϊ�㣬���в�������򻯰渴����a+bi��*��c+di)
			image_real.at<float>(i, j) = -image_ima.at<float>(i, j)*ima.at<float>(i, j);
			image_ima.at<float>(i, j) = image_real.at<float>(i, j)*ima.at<float>(i, j);
			//��ԭ��ԭ����
			//image_real.at<float>(i, j) = image_real.at<float>(i, j);
			//image_ima.at<float>(i, j) = image_ima.at<float>(i, j);
		}
	}

	vector[0] = image_real;
	vector[1] = image_ima;
	merge(vector, 2, src_image_dft);

	//6)��任
	fourior_inverser(src_image_dft, real_src_filter, ima_src_filter);
	divide(real_src_filter, real_src_filter.rows*real_src_filter.cols, real_src_filter);
	divide(ima_src_filter, ima_src_filter.rows*ima_src_filter.cols, ima_src_filter);
	//cout<< real_src_filter <<endl;
	//cout<<ima_src_filter<<endl;

	//7)��������
	for (int i = 0; i < real_src_filter.rows; i++)
	{
		for (int j = 0; j < real_src_filter.cols; j++)
		{
			real_src_filter.at<float>(i, j) = real_src_filter.at<float>(i, j)* pow(-1.0, i + j);
		}
	}

	image_cut(real_src_filter, re);
	//demarcate(real_src_filter);

	normalize(real_src_filter, real_src_filter, 1, 0, NORM_MINMAX);
	//real_src_filter.convertTo(real_src_filter, CV_8U);
	//8)�ü�ͼƬ
	imshow("�˲����ͼ:", real_src_filter);

	Laplace_cuda(dft_lena_filter_space, 3, 1);
	//dft_lena_filter_space.convertTo(dft_lena_filter_space, CV_8U);
	
	dft_lena_filter_space.convertTo(dft_lena_filter_space,CV_32F);
	normalize(dft_lena_filter_space, dft_lena_filter_space, 1, 0, NORM_MINMAX);
	
	//demarcate(dft_lena_filter_space);
	imshow("�ռ��˲����ͼ:", dft_lena_filter_space);
	waitKey(0);
}


//4.42����ЧӦ
void bell_frequency() 
{
	filter_ILPF_bell(50);
	//filter_ILPF_test(10);
	//filter_ILPF_test(30);
	//filter_ILPF_test(60);
	//filter_ILPF_test(160);
	//filter_ILPF_test(460);

	//filter_BLPF_test(10,2);
	//filter_BLPF_test(30,2);
	//filter_BLPF_test(60,2);
	//filter_BLPF_test(160,2);
	//filter_BLPF_test(460,2);
}

void chapter4()
{    
	//space2frequency();
	bell_frequency();
}