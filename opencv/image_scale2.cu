#include "image_scale.cuh" 

//��-��ɾ�������Ŵ��ǹ�ȡ������С��Ƿȡ��ʵ��
using namespace std;
using namespace cv;

namespace image_scale2 {
	//����CUDA����
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_neareast_point;//���ڼ������point��ֵ

	//����CUDA����
	cudaArray* cuArray_neareast_point;//���ڼ������point��ֵ

	//ͨ����
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar>();

	//������������Բ�ֵ
	//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
	__global__ void weightAddKerke_neareast_point(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx) / imgw_rato_d;
			float y_des = (tidy) / imgh_rato_d;
			int x1 = (int)floor(x_des); //ȡ�ĸ���ӽ�Ԫ���У����Ͻǵ�Ԫ��
			int y2 = (int)floor(y_des);
			pDstImgData[idx] = tex2D(refTex_neareast_point, x1, y2);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


	//�ġ��ڲ�ֵ��-��ɾ���㷨��Row and column deletion)= ������ֵ�������ߵȼ�
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_RCDeletion;//���ڼ���RCDeletion
	cudaArray* cuArray_RCDeletion;//���ڼ���RCDeletion
	//�㷨�򵥽��ܣ�1��opencv��ͼ�����ؼ��ٵ�ż������2���á���ͷ�����㷨��ԭͼ��Ѱ������
	//��·��ʱ�������ļ�·������mat��û·��ʱ������mat
	Mat RCDeletion(const char* path, Mat lena_o, float x_rato = 2.0, float y_rato = 2.0, int ifhd = 0)
	{
		Mat Lena;
		if (strlen(path) == 0) {
			Lena = lena_o.clone();
		}
		else {
			Lena = imread(path);
		}

		if (ifhd == 0)//���ǻҶ�ͼҪ����ת��
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

		float x_rato_less = x_rato;
		float y_rato_less = y_rato;

		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_RCDeletion.addressMode[0] = cudaAddressModeClamp;
		refTex_RCDeletion.addressMode[1] = cudaAddressModeClamp;
		refTex_RCDeletion.normalized = false;
		refTex_RCDeletion.filterMode = cudaFilterModePoint;
		//��cuArray������
		cudaMallocArray(&cuArray_RCDeletion, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_neareast_point, cuArray_RCDeletion);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_RCDeletion, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_neareast_point << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_neareast_point);
		cudaFree(pDstImgData1);

		//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
		//imshow("cuda_point�����ֵ��", dstImg1);
		return dstImg1.clone();
		//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
		/*waitKey(0);*/
	}

	//�塢ƽ������ƽ�������˲�����
	//��·��ʱ�������ļ�·������mat��û·��ʱ������mat
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_filter_N;//���ڼ���RCDeletion
	cudaArray* cuArray_filter_N;//���ڼ���RCDeletion

	//��ԭͼ���в�ֵ���������Ժ��ͼ��,imgHeight_des_dԭͼ��, imgWidth_des_dԭͼ��,imgh_rato_d ���ų��ȱ���, imgw_rato_d���ſ�ȱ���
	__global__ void weightAddKerke_filter_N(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d, int* filter_cute, float divide)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
			float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
			int x = (int)floor(x_des); //ȡ�ĸ���ӽ�Ԫ���У����Ͻǵ�Ԫ��
			int y = (int)floor(y_des);
			pDstImgData[idx] = (uchar)((tex2D(refTex_filter_N, x - 1, y - 1)*filter_cute[0] + tex2D(refTex_filter_N, x, y - 1)*filter_cute[1] + tex2D(refTex_filter_N, x + 1, y - 1)*filter_cute[2]
				+ tex2D(refTex_filter_N, x - 1, y)*filter_cute[3] + tex2D(refTex_filter_N, x, y)*filter_cute[4] + tex2D(refTex_filter_N, x + 1, y)*filter_cute[5]
				+ tex2D(refTex_filter_N, x - 1, y + 1)*filter_cute[6] + tex2D(refTex_filter_N, x, y + 1)*filter_cute[7] + tex2D(refTex_filter_N, x + 1, y + 1)*filter_cute[8]) / divide);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}

	//ifhd �ж��Ƿ������ͼ��Ϊһ���Ҷ�ͼ���Ǿ�����ת��
	Mat filter_N(const char* path, Mat lena_o, int mode = 0, int ifhd = 0) {
		int* filter_cute;
		cudaMallocManaged((void**)&filter_cute, 3 * 3 * sizeof(uchar));//ͳһ��ַ����Ҫ�Կ�֧��ͳһ��ַ
		float divide = 0.0;

		if (mode == 0) {
			//mode=0 ƽ���˲���
			//[[1��1��1],
			//[1, 1, 1],
			//[1, 1, 1]]
			for (int i = 0; i < 9; i++)
			{
				filter_cute[i] = 1;
			}
			divide = 9.0f;
		}
		else {
			//mode=1
			//[[1��2��1]
			//[2, 4, 2]
			//[1, 2, 1]]
			filter_cute[0] = 1;   //����ط�����Ƕ���һ����
			filter_cute[1] = 2;
			filter_cute[2] = 1;
			filter_cute[3] = 2;
			filter_cute[4] = 4;
			filter_cute[5] = 2;
			filter_cute[6] = 1;
			filter_cute[7] = 2;
			filter_cute[8] = 1;
			divide = 16.0f;
		}

		Mat Lena;
		if (strlen(path) == 0) {
			Lena = lena_o.clone();
		}
		else {
			Lena = imread(path);
		}

		if (ifhd == 0)//���ǻҶ�ͼҪ����ת��
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

		float x_rato_less = 1;//����ͼ�����ز���
		float y_rato_less = 1;//Ϊ�˼��ٴ���Ķ�����ʵӦ�ü򻯵�

		int imgWidth_src = Lena.cols;//ԭͼ���
		int imgHeight_src = Lena.rows;//ԭͼ���
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	   //����1��������
		cudaError_t t;
		refTex_filter_N.addressMode[0] = cudaAddressModeClamp;
		refTex_filter_N.addressMode[1] = cudaAddressModeClamp;
		refTex_filter_N.normalized = false;
		refTex_filter_N.filterMode = cudaFilterModePoint;

		//��cuArray������
		cudaMallocArray(&cuArray_filter_N, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_filter_N, cuArray_filter_N);
		//�������ݵ�cudaArray
		t = cudaMemcpyToArray(cuArray_filter_N, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//��������Ժ���cpu��ͼ��
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//��С

		//��������Ժ���cuda�ϵ�ͼ��
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_filter_N << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less, filter_cute, divide);
		cudaDeviceSynchronize();

		//��GPU����������ݵ�CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_neareast_point);
		cudaFree(pDstImgData1);
		//namedwindow("cuda_point�˲���", window_normal);
		imshow("cuda_point�˲���", dstImg1);
		return dstImg1.clone();
	}

	int image_scale2()
	{   //δ��ԭͼ�����˲�����
		Mat input = imread("C:/Users/Administrator/Desktop/lena.jpg");
		Mat mat_sx = RCDeletion("", input, 0.5, 0.5, 0);//��С=Ƿ����
		mat_sx = RCDeletion("", mat_sx, 2, 2, 1);//����С���ͼ��ԭ��ȥ
		cvtColor(input, input, COLOR_BGR2GRAY);

		imshow("ԭͼ��", input);//ԭͼ
		imshow("���˲���ԭͼ��", mat_sx);

		//��ԭͼ�����˲�����
		Mat input1 = filter_N("C:/Users/Administrator/Desktop/lena.jpg", input, 1);
		Mat mat_sx1 = RCDeletion("", input1, 0.5, 0.5, 1);//��С=Ƿ����
		mat_sx1 = RCDeletion("", mat_sx1, 2, 2, 1);//����С���ͼ��ԭ��ȥ
		imshow("���˲���ԭͼ��", mat_sx1);
		cout << "----------------------------------" << endl;
		waitKey(0);
		return 0;
	}
}