#include <opencv2/highgui.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;


cv::Mat turnDataToRGBMat_kkk(int height, int width, uchar* frame_data)
{
	cv::Mat image(height, width, CV_8UC3);
	uchar* pxvec;
	int count = 0;
	for (int row = 0; row < height; row++) {
		pxvec = image.ptr<uchar>(row);
		for (int col = 0; col < width; col++) {
			for (int c = 0; c < 3; c++) {
				pxvec[col * 3 + c] = frame_data[count];
				count++;
			}
		}
	}
	return image;
}

cv::Mat turnDataToGrayMat_kkk(int height, int width, uchar* frame_data)
{
	cv::Mat image(height, width, CV_8UC1);
	uchar* pxvec;
	int count = 0;
	for (int row = 0; row < height; row++) {
		pxvec = image.ptr<uchar>(row);
		for (int col = 0; col < width; col++) {
			pxvec[col] = frame_data[count];
			count++;
		}
	}

	return image;
}

void release_kkk(uchar* data)
{
	delete[] data;
}

uchar* turnGrayMatToData_kkk(cv::Mat img)
{
	uchar* buffer = new uchar[img.rows * img.cols];
	uchar* pxvec;
	int count = 0;
	for (int row = 0; row < img.rows; row++) {
		pxvec = img.ptr<uchar>(row);
		for (int col = 0; col < img.cols; col++) {
			buffer[count] = pxvec[col];
			count++;
		}
	}
	return buffer;
}

uchar* turnTowGrayMatToData_kkk(cv::Mat img0, cv::Mat img1)
{
	uchar* buffer = new uchar[img0.rows * img0.cols + img1.rows * img1.cols];
	uchar* pxvec;
	int count = 0;
	for (int row = 0; row < img0.rows; row++) {
		pxvec = img0.ptr<uchar>(row);
		for (int col = 0; col < img0.cols; col++) {
			buffer[count] = pxvec[col];
			count++;
		}
	}
	for (int row = 0; row < img1.rows; row++) {
		pxvec = img1.ptr<uchar>(row);
		for (int col = 0; col < img1.cols; col++) {
			buffer[count] = pxvec[col];
			count++;
		}
	}
	return buffer;
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

		cv::Mat* result = new cv::Mat[2]{ flowx_n, flowy_n };
		return result;
	}
}

uchar* calulateFlow_kkk(int length0, int width0, uchar* data0, int length1, int width1, uchar* data1)
{
	Mat img0 = turnDataToGrayMat_kkk(length0, width0, data0);
	Mat img1 = turnDataToGrayMat_kkk(length1, width1, data1);

	cv::Mat* flows = getFlow(img0, img1);
	uchar* ret = turnTowGrayMatToData_kkk(flows[0], flows[1]);
	delete[] flows;
	return ret;
}

