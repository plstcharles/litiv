#pragma once

#include <opencv2/features2d/features2d.hpp>

#define LBSP_VALIDATE_KEYPOINTS_INTERNALLY

class LBSP : public cv::DescriptorExtractor {
public:
	//! constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
	LBSP(int threshold=10);
	//! constructor 2, threshold = relative intensity 'similarity' threshold used when computing comparisons
	LBSP(float threshold=0.1f);
	//! load extractor params from the specified file node @@@@ not impl
	virtual void read(const cv::FileNode&);
	//! write extractor params to the specified file storage @@@@ not impl
	virtual void write(cv::FileStorage&) const;
	//! sets the 'reference' image to be used for inter-frame comparisons (note: if none set, will default back to intra-frame)
	virtual void setReference(const cv::Mat& image);
	//! returns the current descriptor size, in bytes
	virtual int descriptorSize() const;
	//! returns the current descriptor data type
	virtual int descriptorType() const;
	//! returns whether the extractor is using a relative threshold or not
	virtual bool isUsingRelThreshold() const;
	//! returns the current relative threshold used for comparisons (-1 = invalid/not used)
	virtual float getRelThreshold() const;
	//! returns the current absolute threshold used for comparisons (-1 = invalid/not used)
	virtual int getAbsThreshold() const;


	//! utility, specifies the pixel size of the pattern used (width and height)
	static const int LBSP_PATCH_SIZE = 5;
	//! utility, specifies the number of bytes per descriptor
	static const int LBSP_DESC_SIZE = 2;
	//! utility, computes the descriptors for a set of keypoints in an image, using refImage as the reference image for comparisons (if left empty, will compute intra-frame)
	static void computeImpl(const cv::Mat& origImage, const cv::Mat& refImage, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int threshold);
	//! utility, computes the descriptors for a set of keypoints in an image, using refImage as the reference image for comparisons (if left empty, will compute intra-frame)
	static void computeImpl(const cv::Mat& origImage, const cv::Mat& refImage, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, float threshold);
#ifdef LBSP_VALIDATE_KEYPOINTS_INTERNALLY
	//! utility function, used to create an image preview of the different descriptors extracted over an area via their keypoint locations
	static void recreateDescImage(int nChannels, int nRows, int nCols, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, cv::Mat& output);
#else
	//! utility function, used to create an image preview of the different descriptors extracted over an area, assuming 1 desc/px
	static void recreateDescImage(int nChannels, int nRows, int nCols, const cv::Mat& descriptors, cv::Mat& output);
#endif
	//! utility function, used to illustrate the difference between two descriptor images
	static void calcDescImgDiff(const cv::Mat& descImg1, const cv::Mat& descImg2, cv::Mat& output);
	//! utility function, used to calculate the absolute difference between two small, unsigned data types
	static inline uchar absdiff(uchar a, uchar b) {
		return a<b?b-a:a-b;
	}

protected:
	virtual void computeImpl(const cv::Mat& origImage, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
	const bool m_bUseRelativeThreshold;
	const float m_fThreshold;
	const int m_nThreshold;
	cv::Mat m_oRefImage;
};

