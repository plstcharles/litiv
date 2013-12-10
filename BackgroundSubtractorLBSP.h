#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "LBSP.h"

//! defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSLBSP_USE_SC_THRS_VALIDATION 1
//! defines whether to use or not the advanced morphological operations
#define BGSLBSP_USE_ADVANCED_MORPH_OPS 1
//! defines whether we should extract inter-LBSP or intra-LBSP descriptors from processed frames
#define BGSLBSP_EXTRACT_INTER_LBSP 1
//! defines whether we should use inter-LBSP or intra-LBSP descriptors in the model
#define BGSLBSP_MODEL_INTER_LBSP 0

//! defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.60f)

/*!
	Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm (abstract version).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLBSP : public cv::BackgroundSubtractor {
public:
	//! full constructor used to intialize an 'absolute' LBSP-based background subtractor
	explicit BackgroundSubtractorLBSP(size_t nLBSPThreshold, size_t nDescDistThreshold);
	//! full constructor used to intialize a 'relative' LBSP-based background subtractor
	explicit BackgroundSubtractorLBSP(float fLBSPThreshold, size_t nDescDistThreshold);
	//! default destructor
	virtual ~BackgroundSubtractorLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg);
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints)=0;
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=0)=0;
	//! @@@@@@@@@@@@ ????
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const;
	//! returns the keypoints list used for descriptor extraction (note: by default, these are generated from the DenseFeatureDetector class, and the border points are removed)
	virtual std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);


	// ######## DEBUG PURPOSES ONLY ##########
	int nDebugCoordX, nDebugCoordY;

protected:
	//! background model descriptors samples (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDescSamples;
	//! background model keypoints used for LBSP descriptor extraction (specific to the input image size)
	std::vector<cv::KeyPoint> m_voKeyPoints;
	//! input image size
	cv::Size m_oImgSize;
	//! input image channel size
	size_t m_nImgChannels;
	//! input image type
	int m_nImgType;
	//! absolute per-channel descriptor hamming distance threshold
	const size_t m_nDescDistThreshold;
	//! defines if we're using a relative threshold when extracting LBSP features (kept here since we don't keep an LBSP object)
	const bool m_bLBSPUsingRelThreshold;
	//! LBSP absolute internal threshold (kept here since we don't keep an LBSP object)
	const size_t m_nLBSPThreshold;
	//! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
	const float m_fLBSPThreshold;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
};

