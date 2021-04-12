#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudacodec.hpp"
#include <getopt.h>
#include <stdio.h>
#include "opencv2/video/tracking.hpp"
#include <dirent.h>
#include <ctime>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::cuda;


//
//详细参考https://www.delftstack.com/howto/cpp/how-to-get-list-of-files-in-a-directory-cpp/
int countdir(string directory)
{
	struct dirent *de;
	//opendir返回一个指向数组的指针，该数组包含一个以空值终止的字符序列（即C字符串），代表字符串对象的当前值。
	
	DIR *dir = opendir(directory.c_str());
	if(!dir)
	{	
		printf("opendir() failed! Does it exist?\n");
		cout << directory << "\n" << endl;
		return 0;
	}

	unsigned long count=0;
	while(de = readdir(dir))
	{
		++count;
	}
	return count;
}
//调试之用
//Point2f为浮点类型的Point
//cvIsNaN函数判断一个浮点数是不是一个数，如果不符合 IEEE754标准则返回1， 否则返回0
//该函数返回 x (x为浮点数)的绝对值。
inline bool isFlowCorrect(Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

//Vec3b简单而言就是一个uchar类型的，长度为3的vector向量。
//由于在OpenCV中，使用imread读取到的Mat图像数据，都是用uchar类型的数据存储，对于RGB三通道的图像，每个点的数据都是一个Vec3b类型的数据。
//调试之用
static Vec3b computeColor(float fx, float fy)
{
	static bool first = true;

	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow
	//  than between yellow and green)
	// 颜色过渡的相对长度:
    // 这些选择基于感知相似性
    // 可以区分更多的红色和黄色
    // 介于黄色和绿色之间)
	const int RY = 15;
	const int YG = 6;
	const int GC = 4;
	const int CB = 11;
	const int BM = 13;
	const int MR = 6;
	const int NCOLS = RY + YG + GC + CB + BM + MR;
	static Vec3i colorWheel[NCOLS];

	if (first)
	{
		int k = 0;

		for (int i = 0; i < RY; ++i, ++k)
			colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

		for (int i = 0; i < YG; ++i, ++k)
			colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

		for (int i = 0; i < GC; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

		for (int i = 0; i < CB; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

		for (int i = 0; i < BM; ++i, ++k)
			colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

		for (int i = 0; i < MR; ++i, ++k)
			colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

		first = false;
	}

	const float rad = sqrt(fx * fx + fy * fy);
	const float a = atan2(-fy, -fx) / (float)CV_PI;

	const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	const int k0 = static_cast<int>(fk);
	const int k1 = (k0 + 1) % NCOLS;
	const float f = fk - k0;

	Vec3b pix;

	for (int b = 0; b < 3; b++)
	{
		const float col0 = colorWheel[k0][b] / 255.0f;
		const float col1 = colorWheel[k1][b] / 255.0f;

		float col = (1 - f) * col0 + f * col1;

		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range

		pix[2 - b] = static_cast<uchar>(255.0 * col);
	}

	return pix;
}
//调试之用
static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
	dst.create(flowx.size(), CV_8UC3);
	dst.setTo(Scalar::all(0));

	// determine motion range:
	float maxrad = maxmotion;

	if (maxmotion <= 0)
	{
		maxrad = 1;
		for (int y = 0; y < flowx.rows; ++y)
		{
			for (int x = 0; x < flowx.cols; ++x)
			{
				Point2f u(flowx(y, x), flowy(y, x));

				if (!isFlowCorrect(u))
					continue;

				maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
			}
		}
	}

	for (int y = 0; y < flowx.rows; ++y)
	{
		for (int x = 0; x < flowx.cols; ++x)
		{
			Point2f u(flowx(y, x), flowy(y, x));

			if (isFlowCorrect(u))
				dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
		}
	}
}
//调试之用
static void showFlow(const char* name, const GpuMat& d_flow)
{
	GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]);
	Mat flowy(planes[1]);

	Mat out;
	drawOpticalFlow(flowx, flowy, out, 10);

	imshow(name, out);
}

//#undef 是在后面取消以前定义的宏定义
//cvRound()：返回跟参数最接近的整数值，即四舍五入；
static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
	float lowerBound, float higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flowIn.rows; ++i) {
		for (int j = 0; j < flowIn.cols; ++j) {
			float x = flowIn.at<float>(i, j);
			flowOut.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
		}
	}
#undef CAST
}


int main()
{


// My hacking stuff
//读取数据
ifstream read("/data2/ChaLearn/IsoGD_labels/train_label_1.csv");

string line;

while (std::getline(read, line))
{
	cout << line << endl;
}




// My hacking ends


	float minU = 0;
	float maxU = 0;
	float minV = 0;
	float maxV = 0;
	int num_frames = 0;
	string base_dir;
	string destination_folder1, destination_folder2;
	string filename1, filename2;
	Mat frame0;
	Mat frame1;
	char newName1[50];
	char newName2[50];

	for (int folder_name = 1; folder_name < 148093; folder_name++) {

		base_dir = "sss/data/jester/20bn-dataset/20bn-jester-v1/" + std::to_string(folder_name) + "/";
				
		destination_folder1 = "sss/data/jester/20bn-dataset/flow_abs_max/u/" + std::to_string(folder_name) + "/";
		destination_folder2 = "sss/data/jester/20bn-dataset/flow_abs_max/v/" + std::to_string(folder_name) + "/";
		//不知道为何减2(函数算出来的始终多二,且1不能有中文路径,会识别错误)
		//countdir 算出目录里文件数量
		num_frames = countdir(base_dir) - 2; // omit the '.' and '..' dirs
		cout << "Frame number in the dir '" << base_dir << "' :" << num_frames << endl;
		
		for (int fr_index = 2; fr_index <= num_frames; fr_index++) {
			//填充五位数,不满右补0
			sprintf(newName1, "%05d.jpg", fr_index - 1);
			sprintf(newName2, "%05d.jpg", fr_index);

			filename1 = base_dir  + newName1;
			filename2 = base_dir + newName2;

			//读入灰度图片
			frame0 = imread(filename1, IMREAD_GRAYSCALE);
			frame1 = imread(filename2, IMREAD_GRAYSCALE);
			
			
			if (frame0.empty())
			{
				cerr << "Can't open image [" << filename1 << "]" << endl;
				return -1;
			}
			if (frame1.empty())
			{
				cerr << "Can't open image [" << filename2 << "]" << endl;
				return -1;
			}

			if (frame1.size() != frame0.size())
			{
				cerr << "Images should be of equal sizes" << endl;
				return -1;
			}

			GpuMat d_frame0(frame0);
			GpuMat d_frame1(frame1);

			//CV_32FC2是指一个32位浮点型双通道矩阵
			GpuMat d_flow(frame0.size(), CV_32FC2);
			//brox光流法
			//https://docs.opencv.org/3.4/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html
			Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
			//Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
			//Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
			//Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

			{
				GpuMat d_frame0f;
				GpuMat d_frame1f;

				d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
				d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);


				//GetTickCount返回（retrieve）从操作系统启动到现在所经过（elapsed）的毫秒数，它的返回值是DWORD
				//用于返回CPU的频率。get Tick Frequency。这里的单位是秒，也就是一秒内重复的次数
				const int64 start = getTickCount();
				//推测d_flow为光流
				//computed flow image that has the same size as 第一个参数 and type CV_32FC2.
				brox->calc(d_frame0f, d_frame1f, d_flow);

				const double timeSec = (getTickCount() - start) / getTickFrequency();
				//cout << "Brox : " << timeSec << " sec" << endl;
				//cout << "Video : " << base_dir << endl;
				//showFlow("Brox", d_flow);

				// Ben ekledim!!

				// Split flow into x and y components in GPU
				//在GPU中将流分割成x和y组件
				GpuMat planes[2];
				//Copies each plane of a multi-channel matrix into an array.
				//将一个多通道矩阵的每个平面复制到一个数组中。
				cuda::split(d_flow, planes);
				Mat flowx(planes[0]);
				Mat flowy(planes[1]);

				double min_u, max_u;
				//minMaxLoc从一个矩阵中找出全局的最大值和最小值。
				//https://blog.csdn.net/jndingxin/article/details/108447110?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.baidujs&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.baidujs
				cv::minMaxLoc(flowx, &min_u, &max_u);
				double min_v, max_v;
				cv::minMaxLoc(flowy, &min_v, &max_v);

				if (max_u > abs(min_u)) {
					maxU = max_u;
				}
				else {
					maxU = abs(min_u); 
				}
						
				if (max_v > abs(min_v)) {
					maxV = max_v;
				}
				else {
					maxV = abs(min_v);
				}

				float max_val;
				if (maxV > maxU) {
					max_val = maxV;
				}
				else {
					max_val = maxU;
				}

				float min_u_f;
				float max_u_f;

				float min_v_f;
				float max_v_f;


				min_u_f = -max_val;
				max_u_f = max_val;

				min_v_f = -max_val;
				max_v_f = max_val;

				//CV_8UC1 是指一个8位无符号整型单通道矩阵
				cv::Mat flowx_n(flowx.rows, flowx.cols, CV_8UC1);
				cv::Mat flowy_n(flowy.rows, flowy.cols, CV_8UC1);


				convertFlowToImage(flowx, flowx_n, min_u_f, max_u_f);
				convertFlowToImage(flowy, flowy_n, min_v_f, max_v_f);





				// Save optical flows (x, y) as jpg images
				//cout << "Writing img files" << endl;
				// 第三个参数说明：const std::vector&类型的params，表示为特定格式保存的参数编码，它有默认值std::vector()，所以一般情况下不需要填写。如果更改的话，对于不同的图片格式，其对应的值不同功能不同，如下：
				// 对于JPEG格式的图片，这个参数表示从0-100的图片质量（CV_IMWRITE_JPEG_QUALITY）,默认值是95.
				// 对于PNG格式的图片，这个参数表示压缩级别（CV_IMWRITE_PNG_COMPRESSION）从0-9.较高的值意味着更小的尺寸和更长的压缩时间而默认值是3.
				// 对于PPM，PGM或PBM格式的图片，这个参数表示一个二进制格式标志（CV_IMWRITE_PXM_BINARY），取值为0或1，而默认值为1.
				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
				compression_params.push_back(95);


				string name1, name2;
				name1 = destination_folder1 + newName1;
				name2 = destination_folder2 + newName1;
				
				//该函数是把程序中的Mat类型的矩阵保存为图像到指定位置。
				imwrite(name1, flowx_n, compression_params);
				imwrite(name2, flowy_n, compression_params);

				waitKey();

			}
		}
	}
	return 0;
}
