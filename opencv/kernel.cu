
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "math.h"
#include <device_launch_parameters.h>
#include <sstream>
#include <curand.h>

using namespace std;
using namespace cv;

//����CUDA����
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex1;//���ڼ����Լ��Ĳ�ֵ��ʽ
texture <uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex2;//���ڼ���cuda�����Դ���ֵ

//����CUDA����
cudaArray* cuArray1;
cudaArray* cuArray2;

//ͨ����
cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar>();

//һ��˫���Բ�ֵ����
//��1��˫���Բ�ֵ                     //cuda����x��y�����Ƕ��������Ͻǵ���������ϵ
//                                        ��0��0��----x------>
//(x1,y2)  (x2,y2)
//Q12---------Q22                         |
//     (x,y)                              y
//Q11---------Q21                         |
//(x1,y1) ��x2,y1��

//ע��Q11��Q12��Q22��Q21Ϊ��ӽ��ı���ֵ�㣨x_des,y_des) �ĸ����ص�ֵ����Χ�ڡ�0-255����
//x1,x2,y1,y2Ϊ���ص�Q11��Q12��Q22��Q21����������
__device__ uchar interpolation(int x1, int y2, float x_des, float y_des) {
	int x2 = x1 + 1;
	int y1 = y2 + 1;
	uchar rezult = 0;

	uchar fQ11 = tex2D(refTex1, x1, y1);
	uchar fQ12 = tex2D(refTex1, x1, y2);
	uchar fQ22 = tex2D(refTex1, x2, y2);
	uchar fQ21 = tex2D(refTex1, x2, y1);

	rezult = (uchar)floor((((float)fQ11 / (x2 - x1) * (y2 - y1)) * (x2 - x_des) * (y2 - y_des) + ((float)fQ21 / (x2 - x1) * (y2 - y1)) * (x_des - x1) * (y2 - y_des)
		+ ((float)fQ12 / (x2 - x1) * (y2 - y1)) * (x2 - x_des) * (y_des - y1) + ((float)fQ22 / (x2 - x1) * (y2 - y1)) * (x_des - x1) * (y_des - y1)));

	return rezult;
}

//�����д��ֵ�㷨
//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
__global__ void weightAddKerkel(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
		float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
		int x1 = (int)floor(x_des); //ȡ�ĸ���ӽ�Ԫ���У����Ͻǵ�Ԫ��
		int y2 = (int)floor(y_des);
		pDstImgData[idx] = interpolation(x1, y2, x_des, y_des);
		//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
	}
}

//cuda����tex2D��ֵ
//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
__global__ void weightAddKerkel_cuda(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
		float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
		pDstImgData[idx] = tex2D(refTex2, x_des + 0.5, y_des + 0.5) * 255;
		// printf("value=%u,%f,%f \n", pDstImgData[idx], x_des, y_des);
	}
}

//�ж���block=goalval,mark��¼
__global__ void compareKernel(int blockDim_x, const uchar *a, const uchar *b, int* mark, long length)
{
	extern __shared__ int sum[];

	//grid=dim1,block=dim1
	int step = gridDim.x*blockDim.x;//total size
	int index = blockDim.x*blockIdx.x + threadIdx.x;

	int totalif = 0;

	for (long i = index; i < length; i = i + step)
	{
		if (a[i] == b[i]) {
			totalif = totalif + 1;
		}
	}

	sum[threadIdx.x] = totalif;

	__syncthreads();

	if (threadIdx.x == 0) {
		int sum_thread = 0;
		for (int i = 0; i < blockDim_x; i = i + 1)
		{
			sum_thread = sum_thread + sum[i];
		}
		atomicAdd((int*)mark, sum_thread);
	}
}


//bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
//	if (mat1.empty() && mat2.empty()) {
//		return true;
//	}
//	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
//		mat1.channels() != mat2.channels()) {
//		return false;
//	}
//	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
//		return false;
//	}
//	int nrOfElements1 = mat1.total()*mat1.elemSize();
//	if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
//	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
//	return lvRet;
//}

__global__ void cos_sin_sum(float *y, float* A_cos, float* A_sin, int x_end, float x_step, int N_num)
{
	float pi = 3.1415926f;
	int step = gridDim.x*blockDim.x;
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int N = x_end / x_step;
	for (int i = index; i < N; i = step + i)
	{    //ÿ��f��x[i]����N_num��sin������cos���������γ�
		float x = i * x_step;
		for (int j = 1; j < N_num + 1; j = j + 1)
		{
			y[i] = y[i] + A_sin[i] * sinf(2 * pi*x* j) + A_cos[i] * cosf(2 * pi*x* j);
		}
	}
}

//����һ�������
//float* round_gpu(size_t n = 10, float mean = 5.0, float std = 618.0) {
//
//	//�ڲ��������ܵ������ʹ��ͳһ��ַ����ֵ
//	float* outputPtr = (float*)malloc(n * sizeof(float));
//	cout << outputPtr[1] << endl;
//	curandGenerator_t generator;
//	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//	curandGenerateNormal(generator, outputPtr, n, mean, std);
//	cout << outputPtr[1] << endl;
//	return outputPtr;
//}


//x_endչʾ�����ڸ���,ÿ����x_sptep�����ӵ�cos��sin��������N_num
void plot_cos_sin()
{   //int x_end=4,float x_step=0.005,int N_num=100,char * path="sys.path.append('C:/Users/Administrator/IdeaProjects/Fourier')"
	/*float * a = round_gpu(10, 5.0, 618.0);*/
	//Py_Initialize(); /*��ʼ��python������,���߱�����Ҫ�õ�python������*/
	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString(path);
	//PyObject * pModule = NULL; //shengmingbianliang
	//PyObject * pFunc = NULL;
	//pModule = PyImport_ImportModule("");
	//pFunc = PyObject_GetAttrString(pModule,"print");
	//PyEval_CallObject(pFunc, NULL);
	//Py_Finalize(); /*����python���������ͷ���Դ*/
	//system("pause");
}

int main()
{
	float x_rato_less = 0.5;
	float y_rato_less = 0.5;

	float x_rato_big = 1.5;
	float y_rato_big = 1.5;

	Mat Lena = imread("C:/Users/Administrator/Desktop/lena.jpg");
	Mat Lena1 = Lena.clone();
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	cvtColor(Lena1, Lena1, COLOR_BGR2BGRA);//
	cvtColor(Lena1, Lena1, COLOR_BGRA2GRAY);//
	imshow("ԭͼ",Lena);
	waitKey(0);

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();

	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	int imgWidth_des_big = floor(imgWidth_src * x_rato_big);//�Ŵ�ͼ��
	int imgHeight_des_big = floor(imgHeight_src * y_rato_big);//�Ŵ�ͼ��

	//����1��������
	cudaError_t t;
	refTex1.addressMode[0] = cudaAddressModeClamp;
	refTex1.addressMode[1] = cudaAddressModeClamp;
	refTex1.normalized = false;
	refTex1.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray1, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex1, cuArray1);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray1, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);


	//����2�������� ʹ��cuda�Լ�������
	refTex2.addressMode[0] = cudaAddressModeClamp;
	refTex2.addressMode[1] = cudaAddressModeClamp;
	refTex2.normalized = false;
	refTex2.filterMode = cudaFilterModeLinear;
	//��cuArray������
	cudaMallocArray(&cuArray2, &cuDesc, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex2, cuArray2);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray2, 0, 0, Lena1.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);


	//���ͼ��
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С
	Mat dstImg2 = Mat::zeros(imgHeight_des_big, imgWidth_des_big, CV_8UC1);//�Ŵ�

	uchar* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

	uchar* pDstImgData2 = NULL;
	t = cudaMalloc(&pDstImgData2, imgHeight_des_big * imgWidth_des_big * sizeof(uchar));

	dim3 block(8, 8);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	dim3 grid_big((imgWidth_des_big + block.x - 1) / block.x, (imgHeight_des_big + block.y - 1) / block.y);

	weightAddKerkel_cuda << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
	cudaDeviceSynchronize();
	weightAddKerkel_cuda << <grid_big, block >> > (pDstImgData2, imgHeight_des_big, imgWidth_des_big, y_rato_big, x_rato_big);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
	t = cudaMemcpy(dstImg2.data, pDstImgData2, imgWidth_des_big * imgHeight_des_big * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);

	imshow("��С", dstImg1);
	waitKey(0);
	imshow("�Ŵ�", dstImg2);
	waitKey(0);

	//long size = imgWidth_des * imgHeight_des;
	//int blocksize = 256;
	//int *mark;
	//cudaMallocManaged(&mark, sizeof(int));
	//compareKernel << <1, blocksize, blocksize * sizeof(int) >> > (blocksize, pDstImgData1, pDstImgData2, mark, size);
	//cudaDeviceSynchronize();
	//cout << (float)mark[0] / (float)size*100.0 << endl;

	//��ʾ
	//namedWindow("��С");
	//imshow("��С", dstImg);
   //ostringstream oss;
   //oss << x_rato;
   //string p = oss.str();

	return 0;
}
