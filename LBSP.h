#pragma once

#include <opencv2/features2d/features2d.hpp>

//! defines the default absolute threshold to be used when computing LBSP pattern comparisons
#define LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD 23
//! defines the default relative threshold to be used when computing LBSP pattern comparisons
#define LBSP_DEFAULT_REL_SIMILARITY_THRESHOLD 0.1f

//! defines if the provided keypoints should be validated when passed to 'compute' functions or not; setting to zero might improve performance, but is less safe
#define LBSP_VALIDATE_KEYPOINTS_INTERNALLY 0

/*!
	Local Binary Similarity Pattern (LBSP) feature extractor

	Note 1: both grayscale and RGB/BGR images may be used with this extractor.
	Note 2: using LBSP::compute2(...) is logically equivalent to using DescriptorExtractor::compute(...) followed by LBSP::reshapeDesc(...).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.
 */
class LBSP : public cv::DescriptorExtractor {
public:
	//! utility, specifies the pixel size of the pattern used (width and height)
	static const int PATCH_SIZE = 5;
	//! utility, specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
	static const int DESC_SIZE = 2;

	//! default constructor, falls back to using absolute threshold with default params
	LBSP();
	//! constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
	explicit LBSP(int threshold/*=LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD*/);
	//! constructor 2, threshold = relative intensity 'similarity' threshold used when computing comparisons
	explicit LBSP(float threshold/*=LBSP_DEFAULT_REL_SIMILARITY_THRESHOLD*/);
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
	//! returns whether this extractor is using a relative threshold or not
	virtual bool isUsingRelThreshold() const;
	//! returns the current relative threshold used for comparisons (-1 = invalid/not used)
	virtual float getRelThreshold() const;
	//! returns the current absolute threshold used for comparisons (-1 = invalid/not used)
	virtual int getAbsThreshold() const;

	//! utility function, used to reshape a descriptors matrix to its input image size via their keypoint locations
	static void reshapeDesc(cv::Size size, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, cv::Mat& output);
	//! utility function, used to illustrate the difference between two descriptor images
	static void calcDescImgDiff(const cv::Mat& desc1, const cv::Mat& desc2, cv::Mat& output);
	//! utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
	static void validateKeyPoints(std::vector<cv::KeyPoint>& keypoints, cv::Size imgsize);

	//! similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
	void compute2(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
	//! batch version of LBSP::compute2(const cv::Mat& image, ...), also similar to DescriptorExtractor::compute(const std::vector<cv::Mat>& imageCollection, ...)
	void compute2(const std::vector<cv::Mat>& imageCollection, std::vector<std::vector<cv::KeyPoint> >& pointCollection, std::vector<cv::Mat>& descCollection) const;

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (1-channel version)
	static void computeSingle(const cv::Mat& image, const cv::Mat& ref, const int posX, const int posY, const int threshold, unsigned short& descriptor);
	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (3-channels version)
	static void computeSingle(const cv::Mat& image, const cv::Mat& ref, const int posX, const int posY, const int threshold, unsigned short* descriptors);

protected:
	//! classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
	virtual void computeImpl(const cv::Mat& origImage, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

	const bool m_bUseRelativeThreshold;
	const float m_fThreshold;
	const int m_nThreshold;
	cv::Mat m_oRefImage;
};

