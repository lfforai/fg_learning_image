#include "cufft.cuh"

//����cuda�Դ�cufftģ�� ʵ��ͼ��ĸ���Ҷ�任
using namespace std;
using namespace cv;

//mode=0��ͨƵ�ף�mode=1���Ļ�Ƶ�ף�mode=2���Ļ��������
void cufft(char* path,int mode) {
	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ

	int imgWidth_src = Lena.cols;//ԭͼ��� x
	int imgHeight_src = Lena.rows;//ԭͼ��� y
	
	cout<<"ͼ���x��"<<imgWidth_src<<endl;
	cout<<"ͼ���y��" <<imgHeight_src<<endl;

	int NX = Lena.cols;
	int NY = Lena.rows;
	int length = NX * NY;
		 
	int  BATCH = 1;
	int  NRANK = 2;

    cufftHandle plan;
	cufftComplex *data;
	
	int n[2] = { NX, NY };
	cudaMallocManaged((void**)&data, sizeof(cufftComplex)*NX*NY*BATCH);

	//��ͼ��Ԫ�ظ�ֵ����ֵ��ʵ������
	for (int i = 0;i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			if (mode == 1 || mode == 2)
				data[NX*i + j].x = (float)Lena.data[NX*i + j] * pow(-1.0, i + j);//(0,0)ת�Ƶ���N/2,N/2)����
			else
				data[NX*i + j].x = (float)Lena.data[NX*i + j] ;//�����Ļ�
			data[NX*i + j].y = 0.0;
	/*	    if(i==0 && j==0)
			  printf("aa:%f��%f \n", data[NX*i + j].x, data[NX*i + j].y);*/
		}
	}

	cout<<"--------------ԭͼǰ10������:---------------"<<endl;
	for (int i = 0; i < 10; i++) {
		cout << "i:="<<i<< "��ʵ��:" << data[i].x << "|�鲿:" << data[i].y << endl;
	}
	
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}

	/* Create a 2D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		NULL, 1, NX*NY, // *inembed, istride, idist
		NULL, 1, NX*NY, // *onembed, ostride, odist
		CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}


	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	}

	cout<<"--------------����Ҷ�任:--------------------"<<endl;
	for (int i = 0; i < 10; i++) {
		cout << "i:=" << i << "��ʵ��:" << data[i].x << "|�鲿:" << data[i].y << endl;
	}
	
	//����Ƶ��ͼ�������Ļ������Ļ��������� //�ⲿ����ȫ����cuda����Ŀǰû���������
	cout << "--------------����Ƶ��ͼ:--------------------" << endl;
	float* data_spectrum;
	cudaMallocManaged((void**)&data_spectrum, sizeof(float)*NX*NY);
	
	uchar* data_spectrum_uchar;
	cudaMallocManaged((void**)&data_spectrum_uchar, sizeof(uchar)*NX*NY);

	//����Ƶ��
	float max = 0.0f;
	float min = 10000000000000.0f;
	for (int i = 0; i < NY; i++)
	{   
		for (int j = 0; j < NX; j++)
		{   if (mode==1 || mode==0)
			   data_spectrum[NX*i+j]=sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2));
		    if (mode==2)
		       data_spectrum[NX*i+j] =1+log(sqrt(pow(data[NX*i + j].x, 2) + pow(data[NX*i + j].y, 2)));//����
			if (j == 0 && i == 0) {
				min = data_spectrum[NX*i + j];
				max = data_spectrum[NX*i + j];
			}
			else {
				if (data_spectrum[NX*i + j] < min)
					min = data_spectrum[NX*i + j];
				if (data_spectrum[NX*i + j] > max)
					max = data_spectrum[NX*i + j];
			}
		}
	}

	//��һ���Ժ󣬰�Ƶ�ʱ�Ϊͼ���ʽ
	float max_min = max- min;
	cout<<max_min<<endl;
	cout<<"���ֵ��"<<max<< endl;
	cout<<"��Сֵ��"<<min<< endl;
	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data_spectrum_uchar[NX*i + j]=(uchar)(((data_spectrum[NX*i + j] - min)/ max_min)*255);
		}
	}

	Mat dstImg1 = Mat::zeros(NY,NX, CV_8UC1);//��С
	cudaMemcpy(dstImg1.data, data_spectrum_uchar, NX * NY * sizeof(uchar), cudaMemcpyDefault);
	if(mode==0)
	  imshow("ԭʼƵ��ͼ��", dstImg1);
	if(mode==1)
	  imshow("���Ļ�Ƶ��ͼ��", dstImg1);
	if(mode==2)
	  imshow("���ļ�������Ƶ��ͼ��", dstImg1);

	cout << "--------------����Ҷ���任:--------------------" << endl;
	if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return;
	}
	
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return;
	}

	for (int i = 0; i < NY; i++)
	{
		for (int j = 0; j < NX; j++)
		{
			data[NX*i + j].x = data[NX*i + j].x / length;
			data[NX*i + j].y = data[NX*i + j].y / length;
		}
	}

	for (int i = 0; i < 10; i++) {
		cout << "i:=" << i << "��ʵ��:" << data[i].x << "|�鲿:" << data[i].y << endl;
	}

	cufftDestroy(plan);
	cudaFree(data);
}

void cuffttest(char *path){
	//char *path = "C:/Users/Administrator/Desktop/I.png";
	Mat Lena = imread(path);
	cvtColor(Lena, Lena, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
	imshow("ԭͼ��", Lena);
	cufft(path, 0);//ԭʼƵ��ͼ
	cufft(path, 1);//���Ļ�
	cufft(path, 2);//������
	waitKey(0);
}