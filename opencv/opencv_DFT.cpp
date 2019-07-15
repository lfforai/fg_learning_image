#include "opencv_DFT.h"

//基于opencv的频谱、相位图画法
#define PI2 2*3.141592654

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
