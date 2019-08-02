#pragma once
#include "morphology.cuh"

texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_corrode;//���ڼ���˫���Բ�ֵ	

cudaArray* cuArray_corrode;//����CUDA����

//ͨ����
cudaChannelFormatDesc cuDesc_corrode = cudaCreateChannelDesc<uchar>();


//ͼ��ʴchange center
__device__ uchar change_center(int x, int y, Point_gpu* point_gpu, int len) {
	int x_N;
	int y_N;
	uchar result = 255;
	for (int i = 0; i < len; i++)
	{
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (tex2D(refTex_corrode, x_N, y_N) < 255)//ÿ��point_gpuλ���ϵ����ض���Ҫ��255������õ㽫����ʴ��
		{
			result = 0;
			break;
		}
	}
	return  result;
}

//ͼ��ʴ
__global__ void corrodeKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d,Point_gpu* point_gpu,int len)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			pDstImgData[idx] = (int)change_center(tidx,tidy,point_gpu,len);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


//ͼ������change center
__device__ void change_expand(int* pDstImgData,int x, int y, int imgWidth_des_d, int imgHeight_des_d,Point_gpu* point_gpu,uchar* data, int len) {
	int x_N;
	int y_N;
	int idx;
	for (int i = 0; i < len; i++)
	{   
		x_N = (int)(point_gpu[i].x + x);
		y_N = (int)(point_gpu[i].y + y);
		if (-1 < x_N && x_N < imgWidth_des_d && -1 < y_N && y_N < imgHeight_des_d)
		{   idx = (int)(y_N * imgWidth_des_d + x_N);
		
			 // printf("%u,%u \n", pDstImgData[idx], data[i]);
			if (tex2D(refTex_corrode, x_N, y_N)==255 && data[i] == 0)
			{
				int a =255;
				atomicExch(pDstImgData+idx,a);
				//printf("a: \n");
			   /* pDstImgData[idx] = data[i];*/
			}
			else {
				int a = (int)data[i];
				atomicExch(pDstImgData + idx, a);
				//pDstImgData[idx] =(int) data[i];
			}
		}
	}
}

//ͼ������
__global__ void expandKerkel(int* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, Point_gpu* point_gpu,uchar* data,int len)
{   //printf("threadIdx,x=%d",threadIdx.x);
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
	{
		int idx = tidy * imgWidth_des_d + tidx;
		if(tex2D(refTex_corrode,tidx,tidy)==255)//�����255�����ؼ���Ƿ���Ҫ����
			change_expand(pDstImgData,tidx, tidy, imgWidth_des_d,imgHeight_des_d,point_gpu,data,len);//��Ҫ����
		//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
	}
}

//mode=0 ��ʴ��1=����
void morphology_gpu(char * path,int len,Point_gpu*  point_offset_N,uchar* data ,int mode) {
	se_tpye * se_obj = (se_tpye*)malloc(sizeof(se_tpye));
	se_obj->init(len, point_offset_N,data);

	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	threshold(Lena, Lena, 100, 255, 0);
	image_show(Lena,0.5,"ԭͼ");
	
    int x_rato_less = 1.0;
	int y_rato_less = 1.0;

	int imgWidth_src = Lena.cols;//ԭͼ���
	int imgHeight_src = Lena.rows;//ԭͼ���
	int channels = Lena.channels();


	int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//��Сͼ��
	int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//��Сͼ��

	//����1��������
	cudaError_t t;
	refTex_corrode.addressMode[0] = cudaAddressModeBorder;
	refTex_corrode.addressMode[1] = cudaAddressModeBorder;
	refTex_corrode.normalized = false;
	refTex_corrode.filterMode = cudaFilterModePoint;
	//��cuArray������
	cudaMallocArray(&cuArray_corrode, &cuDesc_corrode, imgWidth_src, imgHeight_src);
	t = cudaBindTextureToArray(refTex_corrode, cuArray_corrode);
	//�������ݵ�cudaArray
	t = cudaMemcpyToArray(cuArray_corrode, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

	//��������Ժ���cpu��ͼ��
	Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_32SC1);//��С

	//��������Ժ���cuda�ϵ�ͼ��
	int* pDstImgData1 = NULL;
	t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(int));

	dim3 block(16, 16);
	dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

	if (mode==0)
	   corrodeKerkel << <grid, block >> > (pDstImgData1,imgHeight_des_less,imgWidth_des_less,se_obj->point_offset,len);
	if (mode==1)
	   expandKerkel << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, se_obj->point_offset, se_obj->data,len);
	cudaDeviceSynchronize();

	//��GPU����������ݵ�CPU
	t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(int)*channels, cudaMemcpyDeviceToHost);
	cudaFree(cuArray_corrode);
	cudaFree(pDstImgData1);
	stringstream ss;
	ss << len;
	string mark;
	ss >> mark;
	string ret = string("��ʴ�Ժ��ͼ") + mark;
	image_show(dstImg1,0.5,ret.c_str());
	//namedWindow("cuda_point�����ֵ��", WINDOW_NORMAL);
	//imshow("��ʴ�Ժ��ͼ��", dstImg1);
	//imwrite("C:/Users/Administrator/Desktop/ͼƬ/Gray_Image0.jpg", dstImg1);
}

//M�ǳ���N�ǿ�
Point_gpu* set_Point_gpu(int M, int N) {
	Point_gpu*  point_offset_N = (Point_gpu*)malloc(sizeof(Point_gpu) * M*N);
	int M_center = (int)M / 2;
	int N_center = (int)N / 2;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			point_offset_N[i*N + j].x = (int)j - N_center;
			point_offset_N[i*N + j].y = (int)i - M_center;
			//cout <<i<<"|"<<j<< endl;
			//cout<<point_offset_N[i*N + j].x <<endl;
			//cout<< point_offset_N[i*N + j].y << endl;
			//cout<<"--------------------------"<<endl;
		}
	}
	return point_offset_N;
}

//��ʴ��
uchar* set_Point_data(int M, int N) {
	uchar*  data = (uchar*)malloc(sizeof(uchar) * M*N);
	int M_center = (int)M / 2;
	int N_center = (int)N / 2;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			data[i*N + j]= 255;
		}
	}
	return data;
}

//������
uchar* set_Point_data_pz(int M, int N) {
	uchar*  data = (uchar*)malloc(sizeof(uchar)* M*N);
    data[0] = 0;
	data[1] = 255;
	data[2] = 0;
	data[3] = 255;
	data[4] = 255;
	data[5] = 255;
	data[6] = 0;
	data[7] = 255;
	data[8] = 0;
	return data;
}

void morphology_test(int M, int N,int mode)
{
	if(mode==0)
	{ Point_gpu* point_offset_N=set_Point_gpu(M,N);
	  uchar* data=set_Point_data(M,N);
	  morphology_gpu("C:/Users/Administrator/Desktop/opencv/m486.png", M*N, point_offset_N,data,0);
	}
	
	if(mode == 1)
	{ Point_gpu * point_offset_N =set_Point_gpu(M, N);;
	  //uchar* data = set_Point_data_pz(M,N);
	  uchar* data = set_Point_data(M, N);
	  morphology_gpu("C:/Users/Administrator/Desktop/opencv/font.png", M*N, point_offset_N, data, 1);
	}
}