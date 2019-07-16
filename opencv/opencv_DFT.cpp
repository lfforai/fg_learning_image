#include "opencv_DFT.h"

//基于opencv的频谱、相位图画法
#define PI2 2*3.141592654

//对比两个mat是否相等，来源于网上
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
	if (mat1.empty() && mat2.empty()) {
		return true;
	}
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
		mat1.channels() != mat2.channels()) {
		return false;
	}
	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
		return false;
	}
	int nrOfElements1 = mat1.total()*mat1.elemSize();
	if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
	return lvRet;
}


int opencv_DFT()
{
	Mat image = imread("C:/Users/Administrator/Desktop/I.png", IMREAD_GRAYSCALE);
	if (image.empty())
		return -1;

	imshow("src", image);
	image.convertTo(image, CV_32FC1);

	/////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
	int oph = getOptimalDFTSize(image.rows);
	int opw = getOptimalDFTSize(image.cols);
	Mat padded;
	copyMakeBorder(image, padded, 0, oph - image.rows, 0, opw - image.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
	Mat complexI;
	merge(temp, 2, complexI);

	dft(complexI, complexI);
	split(complexI, temp);

	Mat amplitude, angle;
	//magnitude(temp[0], temp[1], amplitude);
	//phase(temp[0], temp[1], angle);
	cartToPolar(temp[0], temp[1], amplitude, angle);

	int cx = amplitude.cols / 2;
	int cy = amplitude.rows / 2;
	Mat q0(amplitude, Rect(0, 0, cx, cy));
	Mat q1(amplitude, Rect(cx, 0, cx, cy));
	Mat q2(amplitude, Rect(0, cy, cx, cy));
	Mat q3(amplitude, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	Mat amplitude_src;
	divide(amplitude, oph*opw, amplitude_src);

	imshow("amplitude_src", amplitude_src);

	amplitude += Scalar::all(1);
	log(amplitude, amplitude);
	normalize(amplitude, amplitude, 0, 255, NORM_MINMAX); //归一化 方便显示，和实际数据没有关系
	amplitude.convertTo(amplitude, CV_8U);
	imshow("amplitude", amplitude);

	Mat a0(angle, Rect(0, 0, cx, cy));
	Mat a1(angle, Rect(cx, 0, cx, cy));
	Mat a2(angle, Rect(0, cy, cx, cy));
	Mat a3(angle, Rect(cx, cy, cx, cy));

	Mat tmp_a;
	a0.copyTo(tmp_a);
	a3.copyTo(a0);
	tmp_a.copyTo(a3);

	a1.copyTo(tmp_a);
	a2.copyTo(a1);
	tmp_a.copyTo(a2);

	angle += Scalar::all(1);
	log(angle, angle);
	normalize(angle, angle, 0, 255, NORM_MINMAX); //归一化 方便显示，和实际数据没有关系
	angle.convertTo(angle, CV_8U);

	imshow("angle", angle);
	waitKey(0);
	return 1;
}

//来自微博https ://blog.csdn.net/cyf15238622067/article/details/88231590 
//傅里叶逆变换
Mat fourior_inverser(Mat &src_img,cv::Mat &real_img, cv::Mat &ima_img)
{   
	if (src_img.channels() == 1)
	{
		src_img.convertTo(src_img, CV_32FC1);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
		Mat padded;
		copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));

		Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
		Mat complexI;
		merge(temp, 2, complexI);
		idft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//显示频谱图
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else if (src_img.channels() == 2)
	{
		src_img.convertTo(src_img, CV_32FC2);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);

		Mat temp[] = { Mat::zeros(src_img.size(),CV_32FC1),Mat::zeros(src_img.size(),CV_32FC1) };
		split(src_img, temp);//切分为实部和虚部

		Mat padded_real;//实部按dft要求扩展尺寸
		copyMakeBorder(temp[0], padded_real, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		Mat padded_imal;//虚部按dft要求扩展尺寸
		copyMakeBorder(temp[1], padded_imal, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		temp[0] = padded_real;
		temp[1] = padded_imal;

		Mat complexI;
		//扩展以后再合并
		merge(temp, 2, complexI);
		idft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//分解为实部和虚部
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else {
		cout << "报错,输入的Mat不能为3通道!" << endl;
		return Mat::zeros(2, 2, CV_32FC2);
	}
}

//输入的图像是一个单通道（双通道）的傅里叶变换 
Mat fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	if (src_img.channels() == 1)
	{   
		src_img.convertTo(src_img, CV_32FC1);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
		Mat padded;
		copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));

		Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) };
		Mat complexI;
		merge(temp, 2, complexI);
		dft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//显示频谱图
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else if(src_img.channels() == 2)
   {
		src_img.convertTo(src_img, CV_32FC2);
		///////////////////////////////////////快速傅里叶变换/////////////////////////////////////////////////////
		int oph = getOptimalDFTSize(src_img.rows);
		int opw = getOptimalDFTSize(src_img.cols);
	    
		Mat temp[] = {Mat::zeros(src_img.size(),CV_32FC1),Mat::zeros(src_img.size(),CV_32FC1)};
		split(src_img,temp);//切分为实部和虚部

		Mat padded_real;//实部按dft要求扩展尺寸
		copyMakeBorder(temp[0], padded_real, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi

		Mat padded_imal;//虚部按dft要求扩展尺寸
		copyMakeBorder(temp[1], padded_imal, 0, oph - src_img.rows, 0, opw - src_img.cols,
			BORDER_CONSTANT, Scalar::all(0));//zhi
		
		temp[0] = padded_real;
		temp[1] = padded_imal;

		Mat complexI;
		//扩展以后再合并
		merge(temp, 2, complexI);
		dft(complexI, complexI);//傅里叶变换
		split(complexI, temp);//分解为实部和虚部
		temp[0].copyTo(real_img);
		temp[1].copyTo(ima_img);
		return complexI.clone();
	}
	else {
		cout<<"报错,输入的Mat不能为3通道!"<<endl;
		return Mat::zeros(2, 2, CV_32FC2);
	}
	
}

//图像中心化才能护理
void move_to_center(Mat &center_img)
{
	int cx = center_img.cols / 2;
	int cy = center_img.rows / 2;
	Mat q0(center_img, Rect(0, 0, cx, cy));// Top-Left - Create a ROI per quadrant
	Mat q1(center_img, Rect(cx, 0, cx, cy));// Top-Right
	Mat q2(center_img, Rect(0, cy, cx, cy));// Bottom-Left
	Mat q3(center_img, Rect(cx, cy, cx, cy));// Bottom-Right

	Mat tmp;// swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);// swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

//返回log以后的频谱图[0,255]
void amplitude_common(Mat &center_img){
	center_img.convertTo(center_img,CV_32FC2);
	Mat temp[] = {Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	magnitude(temp[0], temp[1],center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回普通的频谱图[0.255]
void amplitude_log(Mat &center_img) {
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	magnitude(temp[0], temp[1], center_img);
	
	center_img += Scalar::all(1);
	log(center_img, center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回log以后的频谱图[0,255]
void angle_common(Mat &center_img){
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	phase(temp[0], temp[1], center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}

//返回普通的频谱图[0.255]
void angle_log(Mat &center_img){
	center_img.convertTo(center_img, CV_32FC2);
	Mat temp[] = { Mat::zeros(center_img.size(),CV_32FC1),Mat::zeros(center_img.size(),CV_32FC1) };
	split(center_img, temp);//切分为实部和虚部
	phase(temp[0], temp[1], center_img);

	center_img += Scalar::all(1);
	log(center_img, center_img);
	normalize(center_img, center_img, 0, 255, NORM_MINMAX); //归一化方便显示，和实际数据没有关系
	center_img.convertTo(center_img, CV_8U);
}