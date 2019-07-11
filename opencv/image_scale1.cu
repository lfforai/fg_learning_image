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
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_double_linear;//���ڼ���˫���Բ�ֵ
texture <uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_double_linear_cuda;//���ڼ���cuda�����Դ�˫���Բ�ֵ
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_neareast_point;//���ڼ������point��ֵ
texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_Bicubic;//���ڼ���˫�������Բ�ֵ

//����CUDA����
cudaArray* cuArray_double_linear;;//���ڼ���˫���Բ�ֵ
cudaArray* cuArray_double_linea_cuda;//���ڼ���cuda�����Դ�˫���Բ�ֵ
cudaArray* cuArray_neareast_point;//���ڼ������point��ֵ
cudaArray* cuArray_Bicubic;//���ڼ���˫�������Բ�ֵ

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
//x1,x2,y1,y2Ϊ���ص�Q11��Q12��Q22��Q21���������꣬�Լ�д��˫���Բ�ֵ
__device__ uchar interpolation(int x1, int y2, float x_des, float y_des) {
	int x2 = x1 + 1;
	int y1 = y2 + 1;
	uchar rezult = 0;

	uchar fQ11 = tex2D(refTex_double_linear, x1, y1);
	uchar fQ12 = tex2D(refTex_double_linear, x1, y2);
	uchar fQ22 = tex2D(refTex_double_linear, x2, y2);
	uchar fQ21 = tex2D(refTex_double_linear, x2, y1);

	rezult = (uchar)floor((((float)fQ11 / (x2 - x1) * (y2 - y1)) * (x2 - x_des) * (y2 - y_des) + ((float)fQ21 / (x2 - x1) * (y2 - y1)) * (x_des - x1) * (y2 - y_des)
		+ ((float)fQ12 / (x2 - x1) * (y2 - y1)) * (x2 - x_des) * (y_des - y1) + ((float)fQ22 / (x2 - x1) * (y2 - y1)) * (x_des - x1) * (y_des - y1)));

	return rezult;
}

//˫���Բ�ֵ
//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
__global__ void weightAddKerke_double_linear(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
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

//cuda���� �Դ���tex2D˫���Բ�ֵ��ֵ����Ҫʹ��float����
//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
__global__ void weightAddKerkel_double_linear_cuda(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
		float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
		pDstImgData[idx] = tex2D(refTex_double_linear_cuda, x_des + 0.5, y_des + 0.5) * 255;
		//printf("value=%u,%f,%f \n", pDstImgData[idx], x_des, y_des);
	}
}


//������������Բ�ֵ
//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
__global__ void weightAddKerke_neareast_point(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
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
		pDstImgData[idx] = tex2D(refTex_neareast_point, x1, y2);
		//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
	}
}


//����˫���β�ֵBicubi��ֵ����
__device__ float Bicubic(float a, float x) {
	float abs_x = abs(x);
	if (abs_x<= 1.0)
    {
		return (a+2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
	}
	else if(1<abs(x)<2)
	{
		return a * pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
	}
	else { 
		return 0.0f;
	}
}

//˫��������ֵ
__global__ void weightAddKerkel_Bicubic(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
{
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		float a = -0.5;

		int idx = tidy * imgWidth_des_d + tidx;
		float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
		float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
	    int x = (int)floor(x_des); //ȡ�ĸ���ӽ�Ԫ���У����Ͻǵ�Ԫ��
		int y = (int)floor(y_des);
		float u = x_des - x;
		float v = y_des - y;
		//�ҵ�16����
		float k_i0 = Bicubic(a,1.0+u);
		float k_i1 = Bicubic(a, u);
		float k_i2 = Bicubic(a, 1.0-u);
		float k_i3 = Bicubic(a, 2.0-u);
		float k_j0 = Bicubic(a, 1.0+v);
		float k_j1 = Bicubic(a, v);
		float k_j2 = Bicubic(a, 1.0-v);
		float k_j3 = Bicubic(a, 2.0-v);

		pDstImgData[idx] = tex2D(refTex_Bicubic,x - 1, y - 1)*k_i0*k_j0 + tex2D(refTex_Bicubic,x, y - 1)*k_i1*k_j0 + tex2D(refTex_Bicubic,x + 1, y - 1)*k_i2*k_j0 + tex2D(refTex_Bicubic,x + 2, y - 1)*k_i3*k_j0 +
			tex2D(refTex_Bicubic,x - 1, y)*k_i0*k_j1 + tex2D(refTex_Bicubic,x, y)*k_i1*k_j1 + tex2D(refTex_Bicubic,x + 1, y)*k_i2*k_j1 + tex2D(refTex_Bicubic,x + 2, y)*k_i3*k_j1 +
			tex2D(refTex_Bicubic,x - 1, y +1)*k_i0*k_j2 + tex2D(refTex_Bicubic,x, y+1)*k_i1*k_j2 + tex2D(refTex_Bicubic,x + 1, y+1)*k_i2*k_j2 + tex2D(refTex_Bicubic,x + 2, y+1)*k_i3*k_j2 +
			tex2D(refTex_Bicubic,x - 1, y +2)*k_i0*k_j3 + tex2D(refTex_Bicubic,x, y+2)*k_i1*k_j3 + tex2D(refTex_Bicubic,x + 1, y+2)*k_i2*k_j3 + tex2D(refTex_Bicubic,x + 2, y+2)*k_i3*k_j3;
		    //printf("value=%u,%f,%f \n", pDstImgData[idx], x_des, y_des);
	}
}

//mode=0,�����ֵ��ʽ��1=˫���Բ�ֵ��ʽ��2=˫���β�ֵ��ʽ
void image_zooming(char* path = "C:/Users/Administrator/Desktop/lena.jpg",float x_rato=2.0,float y_rato=2.0,int mode=0) 
{   

	float x_rato_less = x_rato;
	float y_rato_less = y_rato;

	if (mode == 0)//�����ֵ��ʽ
	{
		Mat Lena = imread("C:/Users/Administrator/Desktop/lena.jpg");
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	    
		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_neareast_point.addressMode[0] = cudaAddressModeClamp;
		refTex_neareast_point.addressMode[1] = cudaAddressModeClamp;
		refTex_neareast_point.normalized = false;
		refTex_neareast_point.filterMode = cudaFilterModePoint;
		//��cuArray������
		cudaMallocArray(&cuArray_neareast_point, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_neareast_point, cuArray_neareast_point);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_neareast_point, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_neareast_point<< <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_neareast_point);
		cudaFree(pDstImgData1);
		//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
		imshow("cuda_point�����ֵ��", dstImg1);
		imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
		/*waitKey(0);*/
	}

	if (mode == 1) {
		Mat Lena = imread("C:/Users/Administrator/Desktop/lena.jpg");
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_double_linear.addressMode[0] = cudaAddressModeClamp;
		refTex_double_linear.addressMode[1] = cudaAddressModeClamp;
		refTex_double_linear.normalized = false;
		refTex_double_linear.filterMode = cudaFilterModePoint;
		//��cuArray������
		cudaMallocArray(&cuArray_double_linear, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_double_linear, cuArray_double_linear);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_double_linear, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_double_linear << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_double_linear);
		cudaFree(pDstImgData1);
		//namedWindow("�Լ���д˫���Բ�ֵ��", WINDOW_NORMAL);
		imshow("�Լ���д˫���Բ�ֵ��", dstImg1);
		imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image1.jpg", dstImg1);
		//waitKey(0);
	}

	if (mode == 2) {//˫���β�ֵ
		Mat Lena = imread("C:/Users/Administrator/Desktop/lena.jpg");
		cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_Bicubic.addressMode[0] = cudaAddressModeClamp;
		refTex_Bicubic.addressMode[1] = cudaAddressModeClamp;
		refTex_Bicubic.normalized = false;
		refTex_Bicubic.filterMode = cudaFilterModePoint;
		//��cuArray������
		cudaMallocArray(&cuArray_Bicubic, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_Bicubic, cuArray_Bicubic);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_Bicubic, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerkel_Bicubic << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_Bicubic);
		cudaFree(pDstImgData1);
		//namedWindow("˫�����Բ�ֵ��", WINDOW_NORMAL);
		imshow("˫�����Բ�ֵ��", dstImg1);
		imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image2.jpg", dstImg1);
		//waitKey(0);
	}

	if (mode == 3) {//cuda�����Դ�˫���Բ�ֵ
		Mat Lena = imread("C:/Users/Administrator/Desktop/lena.jpg");
		cvtColor(Lena, Lena, COLOR_BGR2BGRA);//
	    cvtColor(Lena, Lena, COLOR_BGRA2GRAY);// 


		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_double_linear_cuda.addressMode[0] = cudaAddressModeClamp;
		refTex_double_linear_cuda.addressMode[1] = cudaAddressModeClamp;
		refTex_double_linear_cuda.normalized = false;
		refTex_double_linear_cuda.filterMode = cudaFilterModeLinear;
		//��cuArray������
		cudaMallocArray(&cuArray_double_linea_cuda, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_double_linear_cuda, cuArray_double_linea_cuda);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_double_linea_cuda, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerkel_double_linear_cuda << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_double_linea_cuda);
		cudaFree(pDstImgData1);
		namedWindow("cuda�Դ�˫���Բ�ֵ��", WINDOW_NORMAL);
		imshow("cuda�Դ�˫���Բ�ֵ��", dstImg1);
		imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image3.jpg", dstImg1);
	/*	waitKey(0);*/
	}

}

int main()
{
	image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 10.0, 10.0, 0);
	image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 10.0, 10.0, 1);
	image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 10.0, 10.0, 2);
	waitKey(0);
	return 0;
}
