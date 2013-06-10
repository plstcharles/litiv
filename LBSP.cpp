#include "LBSP.h"
#include "DistanceUtils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

LBSP::LBSP()
	:	 m_bUseRelativeThreshold(false)
		,m_fThreshold(-1) // unused
		,m_nThreshold(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD)
		,m_oRefImage() {
	CV_DbgAssert(m_nThreshold>0 && m_nThreshold<=UCHAR_MAX);
}

LBSP::LBSP(int threshold)
	:	 m_bUseRelativeThreshold(false)
		,m_fThreshold(-1) // unused
		,m_nThreshold(threshold)
		,m_oRefImage() {
	CV_DbgAssert(m_nThreshold>0 && m_nThreshold<=UCHAR_MAX);
}

LBSP::LBSP(float threshold)
	:	 m_bUseRelativeThreshold(true)
		,m_fThreshold(threshold)
		,m_nThreshold(-1)
		,m_oRefImage() {
	CV_Assert(m_fThreshold>=0 && m_fThreshold<=1);
}

void LBSP::read(const cv::FileNode& fn) {
    // ... = fn["..."];
}

void LBSP::write(cv::FileStorage& fs) const {
    //fs << "..." << ...;
}

void LBSP::setReference(const cv::Mat& img) {
	CV_DbgAssert(img.empty() || img.type()==CV_8UC1 || img.type()==CV_8UC3);
	m_oRefImage = img;
}

int LBSP::descriptorSize() const {
	return DESC_SIZE;
}

int LBSP::descriptorType() const {
	CV_Assert(DESC_SIZE==2); // @@@@ currently only using 16 bit patterns, == unsigned short
	return CV_16U;
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

static inline void lbsp_computeImpl(	const cv::Mat& oInputImg,
										const cv::Mat& oRefImg,
										const std::vector<cv::KeyPoint>& voKeyPoints,
										cv::Mat& oDesc,
										int nThreshold) {
	CV_DbgAssert(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()));
	CV_DbgAssert(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3);
	CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(nThreshold>0 && nThreshold<=UCHAR_MAX);
	const uchar _t = (uchar)nThreshold;
	const int nChannels = oInputImg.channels();
	const int _step_row = oInputImg.step.p[0];
	const uchar* _data = oInputImg.data;
	const uchar* _refdata = oRefImg.empty()?oInputImg.data:oRefImg.data;
	const int nKeyPoints = (int)voKeyPoints.size();
	if(nChannels==1) {
		oDesc.create(nKeyPoints,1,CV_16UC1);
		CV_DbgAssert(oInputImg.step.p[1]==1);
		for(int k=0; k<nKeyPoints; ++k) {
			const int _x = (int)voKeyPoints[k].pt.x;
			const int _y = (int)voKeyPoints[k].pt.y;
			unsigned short& _res = oDesc.at<unsigned short>(k);
#include "LBSP_16bits_dbcross_1ch_abs.i"
		}
	}
	else { //nChannels==3
		oDesc.create(nKeyPoints,1,CV_16UC3);
		CV_DbgAssert(oInputImg.step.p[1]==3);
		for(int k=0; k<nKeyPoints; ++k) {
			const int _x = (int)voKeyPoints[k].pt.x;
			const int _y = (int)voKeyPoints[k].pt.y;
			unsigned short* _res = ((unsigned short*)(oDesc.data + oDesc.step.p[0]*k));
#include "LBSP_16bits_dbcross_3ch_abs.i"
		}
	}
}

static inline void lbsp_computeImpl2(	const cv::Mat& oInputImg,
										const cv::Mat& oRefImg,
										const std::vector<cv::KeyPoint>& voKeyPoints,
										cv::Mat& oDesc,
										int nThreshold) {
	CV_DbgAssert(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()));
	CV_DbgAssert(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3);
	CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(nThreshold>0 && nThreshold<=UCHAR_MAX);
	const uchar _t = (uchar)nThreshold;
	const int nChannels = oInputImg.channels();
	const int _step_row = oInputImg.step.p[0];
	const uchar* _data = oInputImg.data;
	const uchar* _refdata = oRefImg.empty()?oInputImg.data:oRefImg.data;
	const int nKeyPoints = (int)voKeyPoints.size();
	if(nChannels==1) {
		oDesc.create(oInputImg.size(),CV_16UC1);
		CV_DbgAssert(oInputImg.step.p[1]==1);
		for(int k=0; k<nKeyPoints; ++k) {
			const int _x = (int)voKeyPoints[k].pt.x;
			const int _y = (int)voKeyPoints[k].pt.y;
			unsigned short& _res = oDesc.at<unsigned short>(_y,_x);
#include "LBSP_16bits_dbcross_1ch_abs.i"
		}
	}
	else { //nChannels==3
		oDesc.create(oInputImg.size(),CV_16UC3);
		CV_DbgAssert(oInputImg.step.p[1]==3);
		for(int k=0; k<nKeyPoints; ++k) {
			const int _x = (int)voKeyPoints[k].pt.x;
			const int _y = (int)voKeyPoints[k].pt.y;
			unsigned short* _res = ((unsigned short*)(oDesc.data + oDesc.step.p[0]*_y + oDesc.step.p[1]*_x));
#include "LBSP_16bits_dbcross_3ch_abs.i"
		}
	}
}

static inline void lbsp_computeImpl(const cv::Mat& origImage,
									const cv::Mat& refImage,
									const std::vector<cv::KeyPoint>& keypoints,
									cv::Mat& descriptors,
									float _t) {
	// @@@@@@@@@@@@@@@@@@@@@@
	CV_Assert(false);
}

static inline void lbsp_computeImpl2(const cv::Mat& origImage,
									const cv::Mat& refImage,
									const std::vector<cv::KeyPoint>& keypoints,
									cv::Mat& descriptors,
									float _t) {
	// @@@@@@@@@@@@@@@@@@@@@@
	CV_Assert(false);
}

void LBSP::compute2(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    if(image.empty() || keypoints.empty()) {
        descriptors.release();
        return;
    }
#if LBSP_VALIDATE_KEYPOINTS_INTERNALLY
    cv::KeyPointsFilter::runByImageBorder(keypoints,image.size(),PATCH_SIZE/2);
    cv::KeyPointsFilter::runByKeypointSize(keypoints,std::numeric_limits<float>::epsilon());
	CV_DbgAssert(!keypoints.empty());
#endif //LBSP_VALIDATE_KEYPOINTS_INTERNALLY
	if(m_bUseRelativeThreshold)
		lbsp_computeImpl2(image,m_oRefImage,keypoints,descriptors,m_fThreshold);
	else
		lbsp_computeImpl2(image,m_oRefImage,keypoints,descriptors,m_nThreshold);
}

void LBSP::compute2(const std::vector<cv::Mat>& imageCollection, std::vector<std::vector<cv::KeyPoint> >& pointCollection, std::vector<cv::Mat>& descCollection) const {
    CV_Assert(imageCollection.size() == pointCollection.size());
    descCollection.resize(imageCollection.size());
    for(size_t i=0; i<imageCollection.size(); i++)
        compute2(imageCollection[i], pointCollection[i], descCollection[i]);
}

void LBSP::computeSingle(const cv::Mat& oInputImg, const cv::Mat& oRefImg, const int _x, const int _y, const int nThreshold, unsigned short& _res) {
	CV_DbgAssert(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()));
	CV_DbgAssert(oInputImg.type()==CV_8UC1);
	CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(nThreshold>0 && nThreshold<=UCHAR_MAX);
	CV_DbgAssert(oInputImg.step.p[1]==1);
	CV_DbgAssert(_x>=LBSP::PATCH_SIZE/2 && _y>=LBSP::PATCH_SIZE/2);
	CV_DbgAssert(_x<oInputImg.cols-LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-LBSP::PATCH_SIZE/2);
	const uchar _t = (uchar)nThreshold;
	const int _step_row = oInputImg.step.p[0];
	const uchar* _data = oInputImg.data;
	const uchar* _refdata = oRefImg.empty()?oInputImg.data:oRefImg.data;
#include "LBSP_16bits_dbcross_1ch_abs.i"
}

void LBSP::computeSingle(const cv::Mat& oInputImg, const cv::Mat& oRefImg, const int _x, const int _y, const int nThreshold, unsigned short* _res) {
	CV_DbgAssert(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()));
	CV_DbgAssert(oInputImg.type()==CV_8UC3);
	CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(nThreshold>0 && nThreshold<=UCHAR_MAX);
	CV_DbgAssert(oInputImg.step.p[1]==3);
	CV_DbgAssert(_x>=LBSP::PATCH_SIZE/2 && _y>=LBSP::PATCH_SIZE/2);
	CV_DbgAssert(_x<oInputImg.cols-LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-LBSP::PATCH_SIZE/2);
	const uchar _t = (uchar)nThreshold;
	const int _step_row = oInputImg.step.p[0];
	const uchar* _data = oInputImg.data;
	const uchar* _refdata = oRefImg.empty()?oInputImg.data:oRefImg.data;
#include "LBSP_16bits_dbcross_3ch_abs.i"
}

void LBSP::computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
#if LBSP_VALIDATE_KEYPOINTS_INTERNALLY
	cv::KeyPointsFilter::runByImageBorder(keypoints,image.size(),PATCH_SIZE/2);
	CV_DbgAssert(!keypoints.empty());
#endif //LBSP_VALIDATE_KEYPOINTS_INTERNALLY
	if(m_bUseRelativeThreshold)
		lbsp_computeImpl(image,m_oRefImage,keypoints,descriptors,m_fThreshold);
	else
		lbsp_computeImpl(image,m_oRefImage,keypoints,descriptors,m_nThreshold);
}

void LBSP::reshapeDesc(cv::Size size, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, cv::Mat& output) {
	CV_DbgAssert(!keypoints.empty());
	CV_DbgAssert(!descriptors.empty() && descriptors.cols==1);
	CV_DbgAssert(size.width>0 && size.height>0);
	CV_DbgAssert(DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(descriptors.type()==CV_16UC1 || descriptors.type()==CV_16UC3);
	const int nChannels = descriptors.channels();
	const int nKeyPoints = (int)keypoints.size();
	if(nChannels==1) {
		output.create(size,CV_16UC1);
		output = cv::Scalar_<ushort>(0);
		for(int k=0; k<nKeyPoints; ++k)
			output.at<ushort>(keypoints[k].pt) = descriptors.at<ushort>(k);
	}
	else { //nChannels==3
		output.create(size,CV_16UC3);
		output = cv::Scalar_<ushort>(0,0,0);
		CV_DbgAssert(output.step.p[0]==(size_t)size.width*6 && descriptors.step.p[0]==6);
		for(int k=0; k<nKeyPoints; ++k) {
			unsigned short* output_ptr = (unsigned short*)(output.data + output.step.p[0]*(int)keypoints[k].pt.y);
			const unsigned short* desc_ptr = (unsigned short*)(descriptors.data + descriptors.step.p[0]*k);
			const int idx = 3*(int)keypoints[k].pt.x;
			for(int n=0; n<3; ++n)
				output_ptr[idx+n] = desc_ptr[n];
		}
	}
}

void LBSP::calcDescImgDiff(const cv::Mat& desc1, const cv::Mat& desc2, cv::Mat& output) {
	CV_DbgAssert(desc1.size()==desc2.size() && desc1.type()==desc2.type());
	CV_DbgAssert(DESC_SIZE==2); // @@@ also relies on a constant desc size
	CV_DbgAssert(desc1.type()==CV_16UC1 || desc1.type()==CV_16UC3);
	CV_DbgAssert(CV_MAT_DEPTH(desc1.type())==CV_16U);
	CV_DbgAssert(DESC_SIZE*8<=UCHAR_MAX);
	CV_DbgAssert(desc1.step.p[0]==desc2.step.p[0] && desc1.step.p[1]==desc2.step.p[1]);
	const int nScaleFactor = UCHAR_MAX/(DESC_SIZE*8);
	const int nChannels = CV_MAT_CN(desc1.type());
	const int _step_row = desc1.step.p[0];
	if(nChannels==1) {
		output.create(desc1.size(),CV_8UC1);
		CV_DbgAssert(desc1.step.p[1]==2 && output.step.p[1]==1);
		for(int i=0; i<desc1.rows; ++i) {
			const int idx = _step_row*i;
			const unsigned short* desc1_ptr = (unsigned short*)(desc1.data+idx);
			const unsigned short* desc2_ptr = (unsigned short*)(desc2.data+idx);
			for(int j=0; j<desc1.cols; ++j)
				output.at<uchar>(i,j) = nScaleFactor*hdist_ushort_8bitLUT(desc1_ptr[j],desc2_ptr[j]);
		}
	}
	else { //nChannels==3
		output.create(desc1.size(),CV_8UC3);
		CV_DbgAssert(desc1.step.p[1]==6 && output.step.p[1]==3);
		for(int i=0; i<desc1.rows; ++i) {
			const int idx =  _step_row*i;
			const unsigned short* desc1_ptr = (unsigned short*)(desc1.data+idx);
			const unsigned short* desc2_ptr = (unsigned short*)(desc2.data+idx);
			uchar* output_ptr = output.data + output.step.p[0]*i;
			for(int j=0; j<desc1.cols; ++j) {
				for(int n=0;n<3; ++n) {
					const int idx2 = 3*j+n;
					output_ptr[idx2] = nScaleFactor*hdist_ushort_8bitLUT(desc1_ptr[idx2],desc2_ptr[idx2]);
				}
			}
		}
	}
}

void LBSP::validateKeyPoints(std::vector<cv::KeyPoint>& keypoints, cv::Size imgsize) {
	cv::KeyPointsFilter::runByImageBorder(keypoints,imgsize,PATCH_SIZE/2);
}
