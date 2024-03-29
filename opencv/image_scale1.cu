#include "image_scale.cuh"
//最近点插值、双线性插值、双三线性插值实验
using namespace std;
using namespace cv;
namespace image_scale1 {

	//声明CUDA纹理
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_double_linear;//用于计算双线性插值
	texture <uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_double_linear_cuda;//用于计算cuda纹理自带双线性插值
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_neareast_point;//用于计算最近point插值
	texture <uchar, cudaTextureType2D, cudaReadModeElementType> refTex_Bicubic;//用于计算双三次线性插值

	//声明CUDA数组
	cudaArray* cuArray_double_linear;;//用于计算双线性插值
	cudaArray* cuArray_double_linea_cuda;//用于计算cuda纹理自带双线性插值
	cudaArray* cuArray_neareast_point;//用于计算最近point插值
	cudaArray* cuArray_Bicubic;//用于计算双三次线性插值

	//通道数
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar>();

	//一、双线性插值函数
	//（1）双线性插值                     //cuda纹理x，y坐标是顶点在左上角的左手坐标系
	//                                        （0，0）----x------>
	//(x1,y2)  (x2,y2)
	//Q12---------Q22                         |
	//     (x,y)                              y
	//Q11---------Q21                         |
	//(x1,y1) （x2,y1）

	//注：Q11，Q12，Q22，Q21为最接近的被插值点（x_des,y_des) 四个像素点值（范围在【0-255】）
	//x1,x2,y1,y2为像素点Q11，Q12，Q22，Q21的像素坐标，自己写的双线性插值
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

	//双线性插值
	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
	__global__ void weightAddKerke_double_linear(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
			float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
			int x1 = (int)floor(x_des); //取四个最接近元素中，左上角的元素
			int y2 = (int)floor(y_des);
			pDstImgData[idx] = interpolation(x1, y2, x_des, y_des);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}

	//cuda纹理 自带的tex2D双线性插值插值，需要使用float纹理
	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
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


	//二、最近点线性插值
	//对原图进行插值计算缩放以后的图像,imgHeight_des_d原图长, imgWidth_des_d原图宽,imgh_rato_d 缩放长度比例, imgw_rato_d缩放宽度比例
	__global__ void weightAddKerke_neareast_point(uchar* pDstImgData, int imgHeight_des_d, int imgWidth_des_d, float imgh_rato_d, float imgw_rato_d)
	{   //printf("threadIdx,x=%d",threadIdx.x);
		const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
		const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
		if (tidx < imgWidth_des_d && tidy < imgHeight_des_d)
		{
			int idx = tidy * imgWidth_des_d + tidx;
			float x_des = (tidx + 0.5) / imgw_rato_d - 0.5;
			float y_des = (tidy + 0.5) / imgh_rato_d - 0.5;
			int x1 = (int)floor(x_des); //取四个最接近元素中，左上角的元素
			int y2 = (int)floor(y_des);
			pDstImgData[idx] = tex2D(refTex_neareast_point, x1, y2);
			//printf("value=%u,%d,%d,%f,%f \n", pDstImgData[idx], x1, y2, x_des, y_des);
		}
	}

	//三、双三次插值Bicubi插值函数
	__device__ float Bicubic(float a, float x) {
		float abs_x = abs(x);
		if (abs_x <= 1.0)
		{
			return (a + 2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
		}
		else if (1 < abs(x) < 2)
		{
			return a * pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
		}
		else {
			return 0.0f;
		}
	}

	//双三次样插值
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
			int x = (int)floor(x_des); //取四个最接近元素中，左上角的元素
			int y = (int)floor(y_des);
			float u = x_des - x;
			float v = y_des - y;
			//printf("value=%u,%f,%f \n", pDstImgData[idx], x_des, y_des);
			//找到16个点
			float k_i0 = Bicubic(a, 1.0 + u);
			float k_i1 = Bicubic(a, u);
			float k_i2 = Bicubic(a, 1.0 - u);
			float k_i3 = Bicubic(a, 2.0 - u);
			float k_j0 = Bicubic(a, 1.0 + v);
			float k_j1 = Bicubic(a, v);
			float k_j2 = Bicubic(a, 1.0 - v);
			float k_j3 = Bicubic(a, 2.0 - v);
			
			pDstImgData[idx] = (uchar)(tex2D(refTex_Bicubic, x - 1, y - 1)*k_i0*k_j0 + tex2D(refTex_Bicubic, x, y - 1)*k_i1*k_j0 + tex2D(refTex_Bicubic, x + 1, y - 1)*k_i2*k_j0 + tex2D(refTex_Bicubic, x + 2, y - 1)*k_i3*k_j0 +
				tex2D(refTex_Bicubic, x - 1, y)*k_i0*k_j1 + tex2D(refTex_Bicubic, x, y)*k_i1*k_j1 + tex2D(refTex_Bicubic, x + 1, y)*k_i2*k_j1 + tex2D(refTex_Bicubic, x + 2, y)*k_i3*k_j1 +
				tex2D(refTex_Bicubic, x - 1, y + 1)*k_i0*k_j2 + tex2D(refTex_Bicubic, x, y + 1)*k_i1*k_j2 + tex2D(refTex_Bicubic, x + 1, y + 1)*k_i2*k_j2 + tex2D(refTex_Bicubic, x + 2, y + 1)*k_i3*k_j2 +
				tex2D(refTex_Bicubic, x - 1, y + 2)*k_i0*k_j3 + tex2D(refTex_Bicubic, x, y + 2)*k_i1*k_j3 + tex2D(refTex_Bicubic, x + 1, y + 2)*k_i2*k_j3 + tex2D(refTex_Bicubic, x + 2, y + 2)*k_i3*k_j3);
			//printf("value=%u,%f,%f \n", pDstImgData[idx], x_des, y_des);
		}
	}

	//对占用资源教多的函数最好用cuda的最优化函数，测试下gridsize和blocksize是否分配合理
	grid_block_size* bestBlockSize(int used_blocksize,int used_gridsize)
	{
		int blockSize;      // The launch configurator returned block size
		int minGridSize;    // The minimum grid size needed to achieve the
							// maximum occupancy for a full device
							// launch

		grid_block_size* result = (grid_block_size *)malloc(sizeof(grid_block_size));
		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			(void*)weightAddKerkel_Bicubic,
			0,
			used_blocksize);
		result->minGridSize = minGridSize;
		result->blockSize = blockSize;

		if (blockSize == used_blocksize && used_gridsize > minGridSize) {
			cout<<"警告：双三次插值设置的gridsize过大，如果程序出错请调整"<<endl;
		    //waitKey(0);
		}

		if (blockSize <used_blocksize && blockSize*minGridSize<used_gridsize*used_blocksize) {
			cout <<"警告；双三次插值设置的blocksize过大，如果程序出错请调整"<< endl;
			//waitKey(0);
		}
		return result;
	}

	//mode=0,最近插值公式；1=双线性插值公式；2=双三次插值公式
	void image_zooming(char* path = "C:/Users/Administrator/Desktop/lena.jpg", float x_rato = 2.0, float y_rato = 2.0, int mode = 0)
	{

		float x_rato_less = x_rato;
		float y_rato_less = y_rato;

		if (mode == 0)//最近插值公式
		{
			Mat Lena = imread(path);
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

			int imgWidth_src = Lena.cols;//原图像宽
			int imgHeight_src = Lena.rows;//原图像高
			int channels = Lena.channels();
			

			int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
			int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像
			cout<< imgWidth_des_less <<endl;

		   //设置1纹理属性
			cudaError_t t;
			refTex_neareast_point.addressMode[0] = cudaAddressModeClamp;
			refTex_neareast_point.addressMode[1] = cudaAddressModeClamp;
			refTex_neareast_point.normalized = false;
			refTex_neareast_point.filterMode = cudaFilterModePoint;
			//绑定cuArray到纹理
			cudaMallocArray(&cuArray_neareast_point, &cuDesc, imgWidth_src, imgHeight_src);
			t = cudaBindTextureToArray(refTex_neareast_point, cuArray_neareast_point);
			//拷贝数据到cudaArray
			t = cudaMemcpyToArray(cuArray_neareast_point, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

			//输出放缩以后在cpu上图像
			Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

			//输出放缩以后在cuda上的图像
			uchar* pDstImgData1 = NULL;
			t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));
		
			dim3 block(16, 16);
			dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

			weightAddKerke_neareast_point << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
			cudaDeviceSynchronize();

			//从GPU拷贝输出数据到CPU
			t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
			cudaFree(cuArray_neareast_point);
			cudaFree(pDstImgData1);
			//namedWindow("cuda_point最近插值：", WINDOW_NORMAL);
			imshow("cuda_point最近插值：", dstImg1);
			imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image0.jpg", dstImg1);
			/*waitKey(0);*/
		}

		if (mode == 1) //双二次线性插值
		{
			Mat Lena = imread(path);
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

			int imgWidth_src = Lena.cols;//原图像宽
			int imgHeight_src = Lena.rows;//原图像高
			int channels = Lena.channels();

			int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
			int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

		   //设置1纹理属性
			cudaError_t t;
			refTex_double_linear.addressMode[0] = cudaAddressModeClamp;
			refTex_double_linear.addressMode[1] = cudaAddressModeClamp;
			refTex_double_linear.normalized = false;
			refTex_double_linear.filterMode = cudaFilterModePoint;
			//绑定cuArray到纹理
			cudaMallocArray(&cuArray_double_linear, &cuDesc, imgWidth_src, imgHeight_src);
			t = cudaBindTextureToArray(refTex_double_linear, cuArray_double_linear);
			//拷贝数据到cudaArray
			t = cudaMemcpyToArray(cuArray_double_linear, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

			//输出放缩以后在cpu上图像
			Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

			//输出放缩以后在cuda上的图像
			uchar* pDstImgData1 = NULL;
			t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

			dim3 block(8, 8);
			dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

			weightAddKerke_double_linear << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
			cudaDeviceSynchronize();

			//从GPU拷贝输出数据到CPU
			t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
			cudaFree(cuArray_double_linear);
			cudaFree(pDstImgData1);
			//namedWindow("自己编写双线性插值：", WINDOW_NORMAL);
			imshow("自己编写双线性插值：", dstImg1);
			imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image1.jpg", dstImg1);
			//waitKey(0);
		}

		if (mode == 2) {//双三次插值
			Mat Lena = imread(path);
			cvtColor(Lena, Lena, COLOR_BGR2GRAY);//转换为灰度图

			int imgWidth_src = Lena.cols;//原图像宽
			int imgHeight_src = Lena.rows;//原图像高
			int channels = Lena.channels();

			int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
			int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

		   //设置1纹理属性
			cudaError_t t;
			refTex_Bicubic.addressMode[0] = cudaAddressModeClamp;
			refTex_Bicubic.addressMode[1] = cudaAddressModeClamp;
			refTex_Bicubic.normalized = false;
			refTex_Bicubic.filterMode = cudaFilterModePoint;
			//绑定cuArray到纹理
			cudaMallocArray(&cuArray_Bicubic, &cuDesc, imgWidth_src, imgHeight_src);
			t = cudaBindTextureToArray(refTex_Bicubic, cuArray_Bicubic);
			//拷贝数据到cudaArray
			t = cudaMemcpyToArray(cuArray_Bicubic, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

			//输出放缩以后在cpu上图像
			Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

			//输出放缩以后在cuda上的图像
			uchar* pDstImgData1 = NULL;
			t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

			dim3 block(16,16);
			dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);
			
            //cuda资源测试
			grid_block_size* gbs = bestBlockSize(block.x*block.y,grid.x*grid.y);

			weightAddKerkel_Bicubic << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
			cudaDeviceSynchronize();

			//从GPU拷贝输出数据到CPU
			t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
			cudaFree(cuArray_Bicubic);
			cudaFree(pDstImgData1);
			//namedWindow("双三线性插值：", WINDOW_NORMAL);
			imshow("双三线性插值：", dstImg1);
			imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image2.jpg", dstImg1);
			//waitKey(0);
		}

		if (mode == 3) {//cuda纹理自带双线性插值
			Mat Lena = imread(path);
			cvtColor(Lena, Lena, COLOR_BGR2BGRA);//
			cvtColor(Lena, Lena, COLOR_BGRA2GRAY);// 


			int imgWidth_src = Lena.cols;//原图像宽
			int imgHeight_src = Lena.rows;//原图像高
			int channels = Lena.channels();

			int imgWidth_des_less = floor(imgWidth_src * x_rato_less);//缩小图像
			int imgHeight_des_less = floor(imgHeight_src * y_rato_less);//缩小图像

		   //设置1纹理属性
			cudaError_t t;
			refTex_double_linear_cuda.addressMode[0] = cudaAddressModeClamp;
			refTex_double_linear_cuda.addressMode[1] = cudaAddressModeClamp;
			refTex_double_linear_cuda.normalized = false;
			refTex_double_linear_cuda.filterMode = cudaFilterModeLinear;
			//绑定cuArray到纹理
			cudaMallocArray(&cuArray_double_linea_cuda, &cuDesc, imgWidth_src, imgHeight_src);
			t = cudaBindTextureToArray(refTex_double_linear_cuda, cuArray_double_linea_cuda);
			//拷贝数据到cudaArray
			t = cudaMemcpyToArray(cuArray_double_linea_cuda, 0, 0, Lena.data, imgWidth_src * imgHeight_src * sizeof(uchar), cudaMemcpyHostToDevice);

			//输出放缩以后在cpu上图像
			Mat dstImg1 = Mat::zeros(imgHeight_des_less, imgWidth_des_less, CV_8UC1);//缩小

			//输出放缩以后在cuda上的图像
			uchar* pDstImgData1 = NULL;
			t = cudaMalloc(&pDstImgData1, imgHeight_des_less * imgWidth_des_less * sizeof(uchar));

			dim3 block(8, 8);
			dim3 grid((imgWidth_des_less + block.x - 1) / block.x, (imgHeight_des_less + block.y - 1) / block.y);

			weightAddKerkel_double_linear_cuda << <grid, block >> > (pDstImgData1, imgHeight_des_less, imgWidth_des_less, y_rato_less, x_rato_less);
			cudaDeviceSynchronize();

			//从GPU拷贝输出数据到CPU
			t = cudaMemcpy(dstImg1.data, pDstImgData1, imgWidth_des_less * imgHeight_des_less * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);
			cudaFree(cuArray_double_linea_cuda);
			cudaFree(pDstImgData1);
			namedWindow("cuda自带双线性插值：", WINDOW_NORMAL);
			imshow("cuda自带双线性插值：", dstImg1);
			imwrite("C:/Users/Administrator/Desktop/图片/Gray_Image3.jpg", dstImg1);
			/*	waitKey(0);*/
		}

	}

	int image_scale1()
	{
		image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 5.0, 5.0, 0);
		image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 5.0, 5.0, 1);
		image_zooming("C:/Users/Administrator/Desktop/lena1.jpg", 5.0, 5.0, 2);
		waitKey(0);
		return 0;
	}
}