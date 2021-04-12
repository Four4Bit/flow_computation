#include <iostream>
#include <fstream>
#include <filesystem>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudacodec.hpp"
#include "include/getopt.h"
#include <stdio.h>
#include "opencv2/video/tracking.hpp"
#include "include/dirent.h"
#include <ctime>
#include <sstream>
#include "optical_flow.h"

namespace fs = std::filesystem;

using namespace std;
using namespace cv;
using namespace cv::cuda;


int countdir(string directory)
{
	struct dirent* de;
	DIR* dir = opendir(directory.c_str());
	if (!dir)
	{
		printf("opendir() failed! Does it exist?\n");
		cout << directory << "\n" << endl;
		return 0;
	}

	unsigned long count = 0;
	while (de = readdir(dir))
	{
		++count;
	}
	return count;
}

inline bool isFlowCorrect(Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
	static bool first = true;

	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow
	//  than between yellow and green)
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

static void convertFlowToImage(const Mat& flowIn, Mat& flowOut,
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


// $(SolutionDir)$(Platform)\$(Configuration)\  原输出路径
int main()
{
	string projectPath = "E:/sourse/python/MFF-GestureRecognition/MFF-pytorch/";
	string datasetName = "jester";
	string datasetPath = projectPath + "datasets/" + datasetName;

	datasetPath = "F:/My_Resources/datasets/GestureRecognition_fwwb/dataset";

	// 读入标签
	//string labelsPath = projectPath + "datasets/jester-v1/jester-v1-labels.csv";
 //   ifstream read(labelsPath);

 //   string line;

 //   while (std::getline(read, line))
 //   {
 //       cout << line << endl;
 //   }

	// 由于数据太多，实现断点继续功能
	string checkPointPath = datasetPath + "/flow/.checkPoint";
	int startPoint = 1;
	int checkFrequency = 10;  // 缓存检查点频率
	if (fs::exists(checkPointPath)) {
		//shared_ptr<ifstream> checkPointIn;
		//checkPointIn = make_shared<ifstream>(checkPointPath, ios::in);
		ifstream checkPointIn(checkPointPath, ios::in);
		if (checkPointIn.is_open()) {
			checkPointIn >> startPoint;
		}
		checkPointIn.close();
	}

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

	// 遍历所有文件夹
	int floder_num = 11041;
	for (int folder_name = startPoint; folder_name <= floder_num; folder_name++) {

		base_dir = datasetPath + "/rgb/" + std::to_string(folder_name) + "/";

		destination_folder1 = datasetPath + "/flow/u/" + std::to_string(folder_name) + "/";
		destination_folder2 = datasetPath + "/flow/v/" + std::to_string(folder_name) + "/";
		num_frames = countdir(base_dir) - 2; // omit the '.' and '..' dirs
		cout << "Frame number in the dir '" << base_dir << "' :" << num_frames << endl;

		for (int fr_index = 2; fr_index <= num_frames; fr_index++) {
			sprintf_s(newName1, "%05d.jpg", fr_index - 1);
			sprintf_s(newName2, "%05d.jpg", fr_index);

			filename1 = base_dir + newName1;
			filename2 = base_dir + newName2;

			frame0 = imread(filename1, IMREAD_GRAYSCALE);
			frame1 = imread(filename2, IMREAD_GRAYSCALE);

			//while (frame0.empty())
						//{
						//        frame0 = imread(filename1, IMREAD_GRAYSCALE);
						//}
						//while (frame1.empty())
						//{
						//        frame1 = imread(filename2, IMREAD_GRAYSCALE);
						//}


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

			GpuMat d_flow(frame0.size(), CV_32FC2);

			Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
			//Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
			//Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
			//Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

			{
				GpuMat d_frame0f;
				GpuMat d_frame1f;

				d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
				d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

				const int64 start = getTickCount();

				brox->calc(d_frame0f, d_frame1f, d_flow);

				const double timeSec = (getTickCount() - start) / getTickFrequency();
				//cout << "Brox : " << timeSec << " sec" << endl;
				//cout << "Video : " << base_dir << endl;
				//showFlow("Brox", d_flow);

				// Ben ekledim!!

				// Split flow into x and y components in GPU
				GpuMat planes[2];
				cuda::split(d_flow, planes);
				Mat flowx(planes[0]);
				Mat flowy(planes[1]);

				double min_u, max_u;
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

				/*if (max_u > maxU) {
					maxU = max_u;
				}
				if (min_u < minU){
					minU = min_u;
				}

				if (max_v > maxV) {
					maxV = max_v;
				}
				if (min_v < minV) {
					minV = min_v;
				}*/

				//cout << "Max U: " << max_u << "\t" << "Min U: " << min_u << endl;
				//cout << "Max V: " << max_v << "\t" << "Min V: " << min_v << endl;
				//cout << "Max: " << max_val << endl;
				// Normalize optical flows in range [0, 255]
				//Mat flowx_n, flowy_n;
				//cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
				//cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);


				cv::Mat flowx_n(flowx.rows, flowx.cols, CV_8UC1);
				cv::Mat flowy_n(flowy.rows, flowy.cols, CV_8UC1);


				convertFlowToImage(flowx, flowx_n, min_u_f, max_u_f);
				convertFlowToImage(flowy, flowy_n, min_v_f, max_v_f);



				// Save optical flows (x, y) as jpg images
				//cout << "Writing img files" << endl;
				vector<int> compression_params;
				compression_params.push_back(IMWRITE_JPEG_QUALITY);
				//compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  // 可能由于版本变化，该字符串找不到定义
				compression_params.push_back(95);


				string name1, name2;
				name1 = destination_folder1 + newName1;
				name2 = destination_folder2 + newName1;
				//cout << name1 << endl;
				//cout << name2 << endl;

				// 注意目标路径需要创建u/v下的所有文件夹才可以向里面保存图片，所以进行检测，若无，会创建
				makeSurePath(destination_folder1);
				makeSurePath(destination_folder2);

				imwrite(name1, flowx_n, compression_params);
				imwrite(name2, flowy_n, compression_params);
				//imwrite(name1, flowx_n);
				//imwrite(name2, flowy_n);



				//imshow("Brox_x.jpg", flowx_n);
				//imshow("Brox_y.jpg", flowy_n);
				waitKey();

			}

			/*{
				const int64 start = getTickCount();

				lk->calc(d_frame0, d_frame1, d_flow);

				const double timeSec = (getTickCount() - start) / getTickFrequency();
				cout << "LK : " << timeSec << " sec" << endl;

				showFlow("LK", d_flow);
			}

			{
				const int64 start = getTickCount();

				farn->calc(d_frame0, d_frame1, d_flow);

				const double timeSec = (getTickCount() - start) / getTickFrequency();
				cout << "Farn : " << timeSec << " sec" << endl;

				showFlow("Farn", d_flow);
			}

			{
				const int64 start = getTickCount();

				tvl1->calc(d_frame0, d_frame1, d_flow);

				const double timeSec = (getTickCount() - start) / getTickFrequency();
				cout << "TVL1 : " << timeSec << " sec" << endl;

				showFlow("TVL1", d_flow);
			}*/

			//imshow("Frame 0", frame0);
			//imshow("Frame 1", frame1);
		}
		// 保存断点
		if (folder_name % checkFrequency == 0) updateCheckPoint(checkPointPath, folder_name + 1);
	}
	return 0;
}

// frame0/1为灰度图
cv::Mat* getFlow(cv::Mat frame0, cv::Mat frame1)
{
	GpuMat d_frame0(frame0);
	GpuMat d_frame1(frame1);

	GpuMat d_flow(frame0.size(), CV_32FC2);

	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

	{
		GpuMat d_frame0f;
		GpuMat d_frame1f;

		d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
		d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

		const int64 start = getTickCount();

		brox->calc(d_frame0f, d_frame1f, d_flow);

		const double timeSec = (getTickCount() - start) / getTickFrequency();

		// Split flow into x and y components in GPU
		GpuMat planes[2];
		cuda::split(d_flow, planes);
		Mat flowx(planes[0]);
		Mat flowy(planes[1]);

		double min_u, max_u;
		cv::minMaxLoc(flowx, &min_u, &max_u);
		double min_v, max_v;
		cv::minMaxLoc(flowy, &min_v, &max_v);

		float maxU = 0, maxV = 0;
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

		float min_u_f = -max_val;
		float max_u_f = max_val;

		float min_v_f = -max_val;
		float max_v_f = max_val;

		cv::Mat flowx_n(flowx.rows, flowx.cols, CV_8UC1);
		cv::Mat flowy_n(flowy.rows, flowy.cols, CV_8UC1);


		convertFlowToImage(flowx, flowx_n, min_u_f, max_u_f);
		convertFlowToImage(flowy, flowy_n, min_v_f, max_v_f);

		vector<int> compression_params;
		compression_params.push_back(IMWRITE_JPEG_QUALITY);
		//compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  // 可能由于版本变化，该字符串找不到定义
		compression_params.push_back(95);

		cv::Mat result[2]{ flowx_n, flowy_n };
		return result;
	}
}

void updateCheckPoint(std::string& checkPointPath, int point)
{
	ofstream checkPointOut(checkPointPath, ios::out | ios::trunc);
	checkPointOut << point;
	checkPointOut.close();
}

void makeSurePath(std::string& destination_folder1)
{
	if (!fs::exists(destination_folder1))
		fs::create_directories(destination_folder1);
}
