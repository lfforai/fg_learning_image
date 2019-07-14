#include "image_scale.cuh" 

//行-列删除法，放大是过取样，缩小是欠取样实验
using namespace std;
using namespace cv;

namespace image_scale2 {
	//声明CUDA纹理
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_neareast_point;//用于计算最近point插值

	//声明CUDA数组
	cudaArray* cuArray_neareast_point;//用于计算最近point插值

	//通道数
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar>();

	//二、最近点线性插值
	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
	__global__ void weightAddKerke_neareast_point(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx) / imgw_rato_d;
			float y_des = (tidy) / imgh_rato_d;
			int x1 = (int)floor(x_des); //取四个最接近元素中，左上角的元素
			int y2 = (int)floor(y_des);
			pDstImgData[idx] = tex2D(refTex_neareast_point, x1, y2);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}


	//四、内插值列-行删除算法（Row and column deletion)= 最近点插值法，两者等价
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_RCDeletion;//用于计算RCDeletion
	cudaArray* cuArray_RCDeletion;//用于计算RCDeletion
	//算法简单介绍：1、opencv把图像像素减少到偶数倍，2、用“回头看”算法在原图中寻找像素
	//有路径时候优先文件路径导入mat，没路径时候输入mat
	Mat RCDeletion(const char* path, Mat lena_o, float x_rato = 2.0, float y_rato = 2.0, int ifhd = 0)
	{
		Mat Lena;
		if (strlen(path) == 0) {
			Lena = lena_o.clone();
		}
		else {
			Lena = imread(path);
		}

		if (ifhd == 0)//不是灰度图要进行转换
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

		float x_rato_less = x_rato;
		float y_rato_less = y_rato;

		int imgWidth_src = Lena.cols;//原图像宽
		int imgHeight_src = Lena.rows;//原图像高
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

	   //设置1纹理属性
		cudaError_t t;
		refTex_RCDeletion.addressMode[0] = cudaAddressModeClamp;
		refTex_RCDeletion.addressMode[1] = cudaAddressModeClamp;
		refTex_RCDeletion.normalized = false;
		refTex_RCDeletion.filterMode = cudaFilterModePoint;
		//绑定cuArray到纹理
		cudaMallocArray(&cuArray_RCDeletion, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_neareast_point, cuArray_RCDeletion);
		//拷贝数据到cudaArray
		t = cudaMemcpyToArray(cuArray_RCDeletion, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//输出放缩以后在cpu上图像
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

		//输出放缩以后在cuda上的图像
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_neareast_point << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
		cudaDeviceSynchronize();

		//从GPU拷贝输出数据到CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_neareast_point);
		cudaFree(pDstImgData1);

		//namedWindow("cuda_point最近插值：", WINDOW_NORMAL);
		//imshow("cuda_point最近插值：", dstImg1);
		return dstImg1.clone();
		//imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
		/*waitKey(0);*/
	}

	//五、平滑线性平滑线性滤波器，
	//有路径时候优先文件路径导入mat，没路径时候输入mat
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_filter_N;//用于计算RCDeletion
	cudaArray* cuArray_filter_N;//用于计算RCDeletion

	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
	__global__ void weightAddKerke_filter_N(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d, int* filter_cute, float divide)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
			float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
			int x = (int)floor(x_des); //取四个最接近元素中，左上角的元素
			int y = (int)floor(y_des);
			pDstImgData[idx] = (uchar)((tex2D(refTex_filter_N, x - 1, y - 1)*filter_cute[0] + tex2D(refTex_filter_N, x, y - 1)*filter_cute[1] + tex2D(refTex_filter_N, x + 1, y - 1)*filter_cute[2]
				+ tex2D(refTex_filter_N, x - 1, y)*filter_cute[3] + tex2D(refTex_filter_N, x, y)*filter_cute[4] + tex2D(refTex_filter_N, x + 1, y)*filter_cute[5]
				+ tex2D(refTex_filter_N, x - 1, y + 1)*filter_cute[6] + tex2D(refTex_filter_N, x, y + 1)*filter_cute[7] + tex2D(refTex_filter_N, x + 1, y + 1)*filter_cute[8]) / divide);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}

	//ifhd 判断是否输入的图像为一个灰度图，是就无需转换
	Mat filter_N(const char* path, Mat lena_o, int mode = 0, int ifhd = 0) {
		int* filter_cute;
		cudaMallocManaged((void**)&filter_cute, 3 * 3 * sizeof(uchar));//统一地址，需要显卡支持统一地址
		float divide = 0.0;

		if (mode == 0) {
			//mode=0 平均滤波器
			//[[1，1，1],
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
			//[[1，2，1]
			//[2, 4, 2]
			//[1, 2, 1]]
			filter_cute[0] = 1;   //这个地方最好是定义一个宏
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

		if (ifhd == 0)//不是灰度图要进行转换
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

		float x_rato_less = 1;//保持图像像素不变
		float y_rato_less = 1;//为了减少代码改动，其实应该简化掉

		int imgWidth_src = Lena.cols;//原图像宽
		int imgHeight_src = Lena.rows;//原图像高
		int channels = Lena.channels();

		int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
		int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

	   //设置1纹理属性
		cudaError_t t;
		refTex_filter_N.addressMode[0] = cudaAddressModeClamp;
		refTex_filter_N.addressMode[1] = cudaAddressModeClamp;
		refTex_filter_N.normalized = false;
		refTex_filter_N.filterMode = cudaFilterModePoint;

		//绑定cuArray到纹理
		cudaMallocArray(&cuArray_filter_N, &cuDesc, imgWidth_src, imgHeight_src);
		t = cudaBindTextureToArray(refTex_filter_N, cuArray_filter_N);
		//拷贝数据到cudaArray
		t = cudaMemcpyToArray(cuArray_filter_N, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

		//输出放缩以后在cpu上图像
		Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

		//输出放缩以后在cuda上的图像
		uchar* pDstImgData1 = NULL;
		t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

		dim3 block(8, 8);
		dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

		weightAddKerke_filter_N << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less, filter_cute, divide);
		cudaDeviceSynchronize();

		//从GPU拷贝输出数据到CPU
		t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
		cudaFree(cuArray_neareast_point);
		cudaFree(pDstImgData1);
		//namedwindow("cuda_point滤波：", window_normal);
		imshow("cuda_point滤波：", dstImg1);
		return dstImg1.clone();
	}

	int image_scale2()
	{   //未对原图进行滤波处理
		Mat input = imread("C:/Users/Administrator/Desktop/lena.jpg");
		Mat mat_sx = RCDeletion("", input, 0.5, 0.5, 0);//缩小=欠采样
		mat_sx = RCDeletion("", mat_sx, 2, 2, 1);//将缩小后的图还原回去
		cvtColor(input, input, COLOR_BGR2GRAY);

		imshow("原图：", input);//原图
		imshow("无滤波还原图：", mat_sx);

		//对原图进行滤波处理
		Mat input1 = filter_N("C:/Users/Administrator/Desktop/lena.jpg", input, 1);
		Mat mat_sx1 = RCDeletion("", input1, 0.5, 0.5, 1);//缩小=欠采样
		mat_sx1 = RCDeletion("", mat_sx1, 2, 2, 1);//将缩小后的图还原回去
		imshow("有滤波还原图：", mat_sx1);
		cout << "----------------------------------" << endl;
		waitKey(0);
		return 0;
	}
}