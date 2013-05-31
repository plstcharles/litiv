#include "LBSP.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

LBSP::LBSP(int threshold)
	:	 m_bUseRelativeThreshold(false)
		,m_fThreshold(-1) // unused
		,m_nThreshold(threshold)
		,m_oRefImage() // invalid ref
{}

LBSP::LBSP(float threshold)
	:	 m_bUseRelativeThreshold(true)
		,m_fThreshold(threshold)
		,m_nThreshold(-1) // unused
		,m_oRefImage() // invalid ref
{}

void LBSP::read(const cv::FileNode& fn) {
    // ... = fn["..."];
}

void LBSP::write(cv::FileStorage& fs) const {
    //fs << "..." << ...;
}

void LBSP::setReference(const cv::Mat& img) {
	m_oRefImage = img;
}

int LBSP::descriptorSize() const {
	return LBSP_DESC_SIZE;
}

int LBSP::descriptorType() const {
	assert(LBSP_DESC_SIZE==2); // @@@@ currently only accepting 16 bit patterns, == unsigned short
	return CV_16UC1;
}

bool LBSP::isUsingRelThreshold() const {
	return m_bUseRelativeThreshold;
}

float LBSP::getRelThreshold() const {
	return m_fThreshold;
}

int LBSP::getAbsThreshold() const {
	return m_nThreshold;
}

void LBSP::computeImpl(	const cv::Mat& origImage,
						const cv::Mat& refImage,
						std::vector<cv::KeyPoint>& keypoints,
						cv::Mat& descriptors,
						int _t) {
	CV_Assert(refImage.empty() || (refImage.size==origImage.size && refImage.type()==origImage.type() && refImage.channels()==origImage.channels()));
	CV_Assert(LBSP_DESC_SIZE==2); // @@@ also relies on a constant desc size
#ifdef LBSP_VALIDATE_KEYPOINTS_INTERNALLY
	removeBorderKeypoints(keypoints,origImage.size(),LBSP_PATCH_SIZE/2);
#endif
	std::vector<cv::Mat> desc_planes(origImage.channels());
	for(int n=0; n<origImage.channels(); ++n)
		desc_planes[n] = cv::Mat(keypoints.size(),1,CV_16UC1);
	std::vector<cv::Mat> img_planes, ref_planes;
	cv::split(origImage,img_planes);
	if(!refImage.empty())
		cv::split(refImage,ref_planes);
	else
		ref_planes = img_planes;
	unsigned short _res;
	for(size_t n=0; n<img_planes.size(); ++n) {
		const uchar* _data = img_planes[n].data;
		const int _step_row = img_planes[n].step[0];
		const int _step_col = img_planes[n].step[1];
		for(size_t k=0; k<keypoints.size(); ++k) {
			if(	keypoints[k].pt.x < LBSP_PATCH_SIZE/2 ||
				keypoints[k].pt.x >= (origImage.cols-LBSP_PATCH_SIZE/2) ||
				keypoints[k].pt.y < LBSP_PATCH_SIZE/2 ||
				keypoints[k].pt.y >= (origImage.rows-LBSP_PATCH_SIZE/2))
				desc_planes[n].at<unsigned short>(k) = USHRT_MAX;
			else {
				const int _x = (int)keypoints[k].pt.x;
				const int _y = (int)keypoints[k].pt.y;
				const uchar _ref = ref_planes[n].data[_step_row*(_y)+_step_col*(_x)];
#include "LBSP16bitsdbcross_abs.i"
				desc_planes[n].at<unsigned short>(k) = _res;
			}
		}
	}
	cv::merge(desc_planes,descriptors);
}

void LBSP::computeImpl(	const cv::Mat& origImage,
						const cv::Mat& refImage,
						std::vector<cv::KeyPoint>& keypoints,
						cv::Mat& descriptors,
						float _t) {
	// @@@@@@@@@@@@@@@@@@@@@@
	CV_Assert(false);
}

void LBSP::computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
	if(m_bUseRelativeThreshold)
		LBSP::computeImpl(image,m_oRefImage,keypoints,descriptors,m_fThreshold);
	else
		LBSP::computeImpl(image,m_oRefImage,keypoints,descriptors,m_nThreshold);
}

#ifdef LBSP_VALIDATE_KEYPOINTS_INTERNALLY
void LBSP::recreateDescImage(int nChannels, int nRows, int nCols, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, cv::Mat& output) {
	CV_Assert(!keypoints.empty());
#else
cv::Mat LBSP::recreateDescImage(int nChannels, int nRows, int nCols, const cv::Mat& descriptors, cv::Mat& output) {
#endif
	CV_Assert(!descriptors.empty() && nChannels>0 && nCols>0 && nRows>0);
	CV_Assert(LBSP_DESC_SIZE==2); // @@@ also relies on a constant desc size
	std::vector<cv::Mat> res_planes(nChannels);
	for(int n=0; n<nChannels; ++n)
#ifdef LBSP_VALIDATE_KEYPOINTS_INTERNALLY
		res_planes[n] = cv::Mat::zeros(nRows,nCols,CV_16UC1);
#else
		res_planes[n] = cv::Mat(nRows,nCols,CV_16UC1);
#endif
	std::vector<cv::Mat> desc_planes;
	cv::split(descriptors,desc_planes);
	for(size_t n=0; n<desc_planes.size(); ++n)
#ifdef LBSP_VALIDATE_KEYPOINTS_INTERNALLY
		for(size_t k=0; k<keypoints.size(); ++k)
			res_planes[n].at<unsigned short>(keypoints[k].pt) = desc_planes[n].at<unsigned short>(k);
#else
		for(int i=0; i<nRows; ++i)
			for(int j=0; j<nCols; ++j)
				res_planes[n].at<unsigned short>(i,j) = desc_planes[n].at<unsigned short>(i+(j*nRows));
#endif
	cv::merge(res_planes, output);
}

void LBSP::calcDescImgDiff(const cv::Mat& descImg1, const cv::Mat& descImg2, cv::Mat& output) {
	CV_Assert(descImg1.size()==descImg2.size() && descImg1.type()==descImg2.type());
	CV_Assert(LBSP_DESC_SIZE==2 && LBSP_DESC_SIZE*8<=UCHAR_MAX); // @@@ also relies on a constant desc size
	CV_Assert(descImg1.type()==CV_16UC1 || descImg1.type()==CV_16UC3);
	int multFact = UCHAR_MAX/(LBSP_DESC_SIZE*8);
	std::vector<cv::Mat> res_planes(descImg1.channels());
	for(int n=0; n<descImg1.channels(); ++n)
		res_planes[n] = cv::Mat(descImg1.size(),CV_8UC1);
	std::vector<cv::Mat> descImg1_planes, descImg2_planes;
	cv::split(descImg1,descImg1_planes);
	cv::split(descImg2,descImg2_planes);
	for(size_t n=0; n<descImg1_planes.size(); ++n)
		for(int i=0; i<descImg1.rows; ++i)
			for(int j=0; j<descImg1.cols; ++j)
				res_planes[n].at<uchar>(i,j) = multFact*cv::normHamming(&descImg1_planes[n].data[descImg1_planes[n].step[0]*i+descImg1_planes[n].step[1]*j],
																		&descImg2_planes[n].data[descImg2_planes[n].step[0]*i+descImg2_planes[n].step[1]*j],
																		LBSP_DESC_SIZE);
	cv::merge(res_planes, output);
}
