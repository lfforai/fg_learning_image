#include "opencv_chapter3.h"
#include "opencv_DFT.h"

//3.2.1 graph inverse
void graph_inverse(Mat& image_graph) 
{ //only deal with  gray graph
  double max, min;
  cv::Point min_loc, max_loc; //max point
  cv::minMaxLoc(image_graph, &min, &max, &min_loc, &max_loc);
  image_graph =Mat::ones(image_graph.size(),CV_8U)*max-image_graph;
  imshow("breast_inverse", image_graph);
}
 
//3.2.2 log_vary
void log_vary(Mat& Lena ,float c)
{   
	//only deal with  gray graph
	Lena.convertTo(Lena, CV_32FC1);
	Mat real_img;
	Mat ima_img;
	Mat Lena_dft = fast_dft(Lena, real_img, ima_img);
	magnitude(real_img, ima_img, Lena_dft);
	Lena_dft += Scalar(1);
    log(Lena_dft,Lena_dft);
	Lena = Lena * c;
	move_to_center(Lena_dft);
	normalize(Lena_dft, Lena_dft, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	Lena_dft.convertTo(Lena_dft, CV_8U);
	imshow("log_vary",Lena_dft);
}


//r数字越高，图像越暗
void  gamma(Mat& Lena, float c,float r) {
	Lena.convertTo(Lena, CV_32F,1.0/255,0);
	pow(Lena,r,Lena);
	Lena = Lena*c;
	Lena.convertTo(Lena, CV_8U,255,0);
	string name =string("gamma") + to_string(Lena.cols);
	imshow(name, Lena);
}

//3.3 统计直方图
void Histogram(Mat& image) {
	Mat hist;
	int channl[1];
	channl[0] = 0;

	int histSize[1];
	histSize[0] = 256;

	float hranges[2];
	const float* ranges[1];

	hranges[0] = 0.0;
	hranges[1] = 256.0;
	ranges[0] = hranges;

	calcHist(&image,
		1,//仅为一个图像的直方图
		channl,//使用的通道
		cv::Mat(),//不使用掩码
		hist,//作为结果的直方图
		1,//这时一维的直方图
		histSize,//箱子数量
		ranges//像素值的范围
	    );
	hist.copyTo(image);
}

//把hist转换为直方图的样子
//来源网上帖子:https://blog.csdn.net/qq_30241709/article/details/78539644
cv::Mat getImageofHistogram(const cv::Mat &hist, int zoom)
{
	double maxVal = 0;
	double minVal = 0;
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

	int histSize = hist.rows;
	cv::Mat histImg(histSize*zoom, histSize*zoom, CV_8U, cv::Scalar(255));
	//设置最高点为90%箱子的个数
	int hpt = static_cast<int>(0.9*histSize);
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		if (binVal > 0)
		{
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h*zoom, histSize*zoom),
				cv::Point(h*zoom, (histSize - intensity)*zoom),
				cv::Scalar(0), zoom);
		}
	}

	return histImg;
}

//直方图概率进行简单图像转换
void hist_converse(Mat &image_src,Mat &hist)
{
	int cols = hist.cols;
	int rows = hist.rows;
	
	Scalar NM=sum(hist);
	hist.convertTo(hist, CV_32F);
	image_src.convertTo(image_src, CV_32F);
	
	//计算转换表
	Mat convers_talbe = Mat(hist.size(), CV_32F);
	float sum=0.0;
	for (int i = 0;i < rows; i++)
	{ sum=sum+hist.at<float>(i, 0);
	  convers_talbe.at<float>(i, 0) =sum;
	}
	convers_talbe = convers_talbe /NM[0]*255;
	convers_talbe.convertTo(convers_talbe,CV_8U);//转换过程中数据是按四舍五入

	//利用映射表替换image_src中的函数
	for(int i = 0; i < image_src.rows; i++)
	  {
	    for(int j = 0; j < image_src.cols; j++)
	      {
		   image_src.at<float>(i, j)=(float)convers_talbe.at<uchar>((uchar)image_src.at<float>(i,j),0);
	      }
	  }
	image_src.convertTo(image_src, CV_8U);
}

//------chapter3_test--------
void chapter3_test() 
{
	//3.2.1 graph inverse
	Mat breast = imread("C:/Users/Administrator/Desktop/opencv/breast.png", IMREAD_GRAYSCALE);
	imshow("breast_原图", breast);
	graph_inverse(breast);

	//3.2.2 log_vary
	Mat Lena = imread("C:/Users/Administrator/Desktop/opencv/Lena.jpg", IMREAD_GRAYSCALE);	
	Mat Lena_print = Lena.clone();
	imshow("Lena_原图", Lena_print);
	log_vary(Lena,1.0);
	
	//3.2.3 gamma
	Mat skeleton = imread("C:/Users/Administrator/Desktop/opencv/skeleton.png", IMREAD_GRAYSCALE);
	imshow("skeleton_原图", skeleton);
	gamma(skeleton,1.0,0.5);

	Mat street = imread("C:/Users/Administrator/Desktop/opencv/street.png", IMREAD_GRAYSCALE);
	imshow("street_原图", street);
	gamma(street, 1.0, 8.0);

	//3.3.1 图像直方图
	Mat blood = imread("C:/Users/Administrator/Desktop/opencv/blood3.png", IMREAD_GRAYSCALE);
	imshow("blood_原图",blood);
	Mat blood_src=blood.clone();
	Histogram(blood);
	Mat hist= blood.clone();
	Mat histmat=getImageofHistogram(blood,1);
	imshow("blood_直方图",histmat);
	hist_converse(blood_src,hist);
	imshow("blood_转化后的直方图",blood_src);

	waitKey(0);
}