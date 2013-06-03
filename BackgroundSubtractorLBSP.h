#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "LBSP.h"

//! defines the default absolute threshold to be used when evaluating if a pixel is foreground or not
#define BGSLBSP_DEFAULT_FG_THRESHOLD (9)
//! defines the extra threshold value (over m_nFGThreshold) a single channel needs so we can automatically consider the whole pixel (and any other channel) as foreground
#define BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD_DIFF (2)
//! defines the default absolute threshold a single channel needs so we can automatically consider the whole pixel (and any other channel) as foreground
#define BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD (BGSLBSP_DEFAULT_FG_THRESHOLD+BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD_DIFF)

/*!
	Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters
	are adjusted automatically). When processing grayscale images, only m_nFGThreshold is
	used to determine if a pixel is foreground/background.

	For more details on the different parameters, go to @@@@@@@@@@@@@@.
 */
class BackgroundSubtractorLBSP : public cv::BackgroundSubtractor {
public:
	//! default constructor (also using default LBSP descriptor extractor params)
	BackgroundSubtractorLBSP();
	//! full constructor based on the absolute LBSP extractor, with algorithm parameters passed as arguments
	BackgroundSubtractorLBSP(	int nDescThreshold,
								int nFGThreshold=BGSLBSP_DEFAULT_FG_THRESHOLD,
								int nFGSCThreshold=BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD
								);
	//! full constructor based on the relative LBSP extractor, with algorithm parameters passed as arguments
	BackgroundSubtractorLBSP(	float fDescThreshold,
								int nFGThreshold=BGSLBSP_DEFAULT_FG_THRESHOLD,
								int nFGSCThreshold=BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD
								);
	//! default destructor
	virtual ~BackgroundSubtractorLBSP();


	// @@@@@ NOTE: might need to mutex the functions below...


	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Size& oFrameSize, int nFrameType);
	//! primary model update function
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=-1.0);
	//! @@@@@@@@@@@@ ????
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background image
	cv::Mat getCurrentBGImage() const;
	//! returns a copy of the latest background descriptors vector (note: these are paired with the model's keypoints list)
	cv::Mat getCurrentBGDescriptors() const;
	//! returns a displayable representation of the model's latest background descriptors vector
	cv::Mat getCurrentBGDescriptorsImage() const;
	//! returns the keypoints list used for descriptor extraction (note: by default, they are generated from the DenseFeatureDetector class, but the border points are removed)
	std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border points and bases itself on the current input size)
	void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);

private:
	//! background image used as the reference image for comparisons
	cv::Mat m_oBGImg;
	//! background descriptors vector used as the reference vector for comparisons
	cv::Mat m_oBGDesc;
	//! contains the current background keypoints used for descriptor extraction
	std::vector<cv::KeyPoint> m_voBGKeyPoints;
	//! contains the current input image size
	cv::Size m_oImgSize;
	//! contains the current input image channel size
	int m_nImgChannels;
	//! contains the current input image type
	int m_nImgType;
	//! absolute descriptor change threshold used for foreground evaluation
	const int m_nFGThreshold;
	//! absolute descriptor single-channel change threshold used for foreground evaluation (used along with m_nFGThreshold when m_nImgChannels>1)
	const int m_nFGSCThreshold;
	//! m_nFGThreshold, adjusted for the current number of channels
	int m_nCurrFGThreshold;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
	//! LBSP feature extractor
	LBSP m_oExtractor;

	bool gotbgmodel; // @@@@@@@@@@@
};

