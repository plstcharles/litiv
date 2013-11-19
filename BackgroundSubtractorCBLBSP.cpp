#include "BackgroundSubtractorCBLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// local define used for debug purposes only
#define DISPLAY_DEBUG_FRAMES 1
// local define used for debug purposes only
#define USE_SAMPLES_DEBUG_STRUCT 0
// local define for the gradient proportion value used in color+grad distance calculations
#define OVERLOAD_GRAD_PROP ((1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))*0.5f)
// local define for the scale factor used to determine very good word matches
#define GOOD_DIST_SCALE_FACTOR 0.8f
// local define for the lword representation update rate
#define LOCAL_WORD_REPRESENTATION_UPDATE_RATE 16
// local define for the lword replacement rate
#define LOCAL_WORD_REPLACEMENT_RATE 8
// local define for 'average' word weight
#define AVERAGE_WORD_WEIGHT 0.225f
// local define for 'high' word weight
#define HIGH_WORD_WEIGHT 0.400f
// local define for the replaceable lword fraction
#define LWORD_REPLACEABLE_FRAC 4
// local define for the amount of weight offset to apply to words, making sure new words aren't always better than old ones
#define LWORD_WEIGHT_OFFSET 20
// local define for the distance threshold used to consider two words similar on initialisation
#define LWORD_DISTANCE_INIT 0.15f



static const int s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const int s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const int s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const int s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorCBLBSP::BackgroundSubtractorCBLBSP(	 float fLBSPThreshold
														,int nInitDescDistThreshold
														,int nInitColorDistThreshold
														,int nLocalWords
														,int nGlobalWords)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_nLocalWords(nLocalWords)
		,m_nLastLocalWordReplaceableIdxs(m_nLocalWords<LWORD_REPLACEABLE_FRAC?1:(m_nLocalWords/LWORD_REPLACEABLE_FRAC))
		,m_nGlobalWords(nGlobalWords)
		,m_nLocalDictionaries(0)
		,m_nCurrWIDSeed(0)
		,m_aapLocalWords(NULL)
		,m_apGlobalWords(NULL) {
	CV_Assert(m_nLocalWords>0 && m_nGlobalWords>0);
	CV_Assert(m_nLastLocalWordReplaceableIdxs>0);
	CV_Assert(m_nColorDistThreshold>0);
}

BackgroundSubtractorCBLBSP::~BackgroundSubtractorCBLBSP() {
	CleanupDictionaries();
}

void BackgroundSubtractorCBLBSP::CleanupDictionaries() {
	if(m_aapLocalWords) {
		for(int d=0; d<m_nLocalDictionaries; ++d) {
			if(m_aapLocalWords[d]) {
				for(int w=0; w<m_nLocalWords; ++w) {
					if(m_aapLocalWords[d][w]) {
						delete m_aapLocalWords[d][w];
					}
				}
				delete[] m_aapLocalWords[d];
			}
		}
		delete[] m_aapLocalWords;
	}
	m_aapLocalWords = NULL;
	if(m_apGlobalWords) {
		for(int w=0; w<m_nGlobalWords; ++w) {
			if(m_apGlobalWords[w]) {
				delete m_apGlobalWords[w];
			}
		}
		delete[] m_apGlobalWords;
	}
	m_apGlobalWords = NULL;
}

void BackgroundSubtractorCBLBSP::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	std::vector<cv::KeyPoint> voNewKeyPoints;
	if(voKeyPoints.empty()) {
		cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
		voNewKeyPoints.reserve(oInitImg.rows*oInitImg.cols);
		oKPDDetector.detect(cv::Mat(oInitImg.size(),oInitImg.type()),voNewKeyPoints);
	}
	else
		voNewKeyPoints = voKeyPoints;
	LBSP::validateKeyPoints(voNewKeyPoints,oInitImg.size());
	CV_Assert(!voNewKeyPoints.empty());
	m_voKeyPoints = voNewKeyPoints;
	CleanupDictionaries();
	m_nLocalDictionaries = oInitImg.cols*oInitImg.rows;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nFrameIndex = 0;
	m_aapLocalWords = new LocalWord**[m_nLocalDictionaries];
	memset(m_aapLocalWords,0,m_nLocalDictionaries*sizeof(LocalWord**));
	m_apGlobalWords = new GlobalWord*[m_nGlobalWords];
	memset(m_apGlobalWords,0,m_nGlobalWords*sizeof(GlobalWord*));
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(1.0f);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(BGSCBLBSP_T_LOWER);
	m_oWeightThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oWeightThresholdFrame = cv::Scalar(0.0f);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanSegmResFrame = cv::Scalar(0.0f);
	m_oTempFGMask.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_last = cv::Scalar_<uchar>(0);
	m_oPureFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC(m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC(m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
#if USE_SAMPLES_DEBUG_STRUCT
	m_voBGColorSamples.resize(m_nLocalWords);
	m_voBGDescSamples.resize(m_nLocalWords);
	for(int w=0; w<m_nLocalWords; ++w) {
		m_voBGColorSamples[w].create(m_oImgSize,CV_8UC(m_nImgChannels));
		m_voBGColorSamples[w] = cv::Scalar_<uchar>::all(0);
		m_voBGDescSamples[w].create(m_oImgSize,CV_16UC(m_nImgChannels));
		m_voBGDescSamples[w] = cv::Scalar_<ushort>::all(0);
	}
#endif
	const int nKeyPoints = (int)m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(int t=0; t<=UCHAR_MAX; ++t) {
			int nCurrLBSPThreshold = (int)(t*m_fLBSPThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			m_nLBSPThreshold_8bitLUT[t]=nCurrLBSPThreshold>UCHAR_MAX?UCHAR_MAX:(uchar)nCurrLBSPThreshold;
		}
		for(int k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert((int)m_oLastColorFrame.step.p[0]==m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const int idx_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const int idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_nLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
		for(int k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert((int)m_oLastColorFrame.step.p[0]==m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const int idx_orig_ldict = m_oLastColorFrame.cols*y_orig + x_orig;
			m_aapLocalWords[idx_orig_ldict] = new LocalWord*[m_nLocalWords];
			for(int w=0; w<m_nLocalWords; ++w) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const int idx_sample_color = m_oLastColorFrame.cols*y_sample + x_sample;
				const int idx_sample_desc = idx_sample_color*2;
				LocalWord_1ch* pCurrLocalWord = new LocalWord_1ch(m_nCurrWIDSeed);
				pCurrLocalWord->nColor = m_oLastColorFrame.data[idx_sample_color];
				pCurrLocalWord->nDesc = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				m_aapLocalWords[idx_orig_ldict][w] = pCurrLocalWord;
#if USE_SAMPLES_DEBUG_STRUCT
				const int idx_orig_color = idx_orig_ldict;
				CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
				const int idx_orig_desc = idx_orig_color*2;
				m_voBGColorSamples[w].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
				*((ushort*)(m_voBGDescSamples[w].data+idx_orig_desc)) = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
#endif //USE_SAMPLES_DEBUG_STRUCT
			}
		}
	}
	else { //m_nImgChannels==3
		for(int t=0; t<=UCHAR_MAX; ++t) {
			int nCurrLBSPThreshold = (int)(t*m_fLBSPThreshold);
			m_nLBSPThreshold_8bitLUT[t]=nCurrLBSPThreshold>UCHAR_MAX?UCHAR_MAX:(uchar)nCurrLBSPThreshold;
		}
		for(int k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert((int)m_oLastColorFrame.step.p[0]==3*m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==3);
			const int idx_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const int idx_desc = idx_color*2;
			for(int c=0; c<3; ++c) {
				int nCurrBGInitColor = oInitImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_nLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
		for(int k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert((int)m_oLastColorFrame.step.p[0]==3*m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==3);
			const int idx_orig_ldict = m_oLastColorFrame.cols*y_orig + x_orig;
			m_aapLocalWords[idx_orig_ldict] = new LocalWord*[m_nLocalWords];
			for(int w=0; w<m_nLocalWords; ++w) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const int idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const int idx_sample_desc = idx_sample_color*2;
				const uchar* const init_color_ptr = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_sample_desc);
				LocalWord_3ch* pCurrLocalWord = new LocalWord_3ch(m_nCurrWIDSeed);
				for(int c=0; c<3; ++c) {
					pCurrLocalWord->anColor[c] = init_color_ptr[c];
					pCurrLocalWord->anDesc[c] = init_desc_ptr[c];
				}
				m_aapLocalWords[idx_orig_ldict][w] = pCurrLocalWord;
#if USE_SAMPLES_DEBUG_STRUCT
				const int idx_orig_color = idx_orig_ldict*3;
				CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
				const int idx_orig_desc = idx_orig_color*2;
				uchar* bg_color_ptr = m_voBGColorSamples[w].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[w].data+idx_orig_desc);
				for(int c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
				}
#endif //USE_SAMPLES_DEBUG_STRUCT
			}
			for(int w1=0; w1<m_nLocalWords; ++w1) {
				for(int w2=w1+1; w2<m_nLocalWords; ++w2) {
					if(m_aapLocalWords[idx_orig_ldict][w1]->distance(m_aapLocalWords[idx_orig_ldict][w2])<LWORD_DISTANCE_INIT)
						++m_aapLocalWords[idx_orig_ldict][w1]->nOccurrences;
				}
			}
		}
	}
	m_bInitialized = true;
}

void BackgroundSubtractorCBLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	++m_nFrameIndex;
	const int nKeyPoints = (int)m_voKeyPoints.size();
#if DISPLAY_DEBUG_FRAMES
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	int debug_ldict_idx = m_oImgSize.width*nDebugCoordY + nDebugCoordX;
	int best_lword_idx;
	std::cout << std::endl << std::endl << "n=" << m_nFrameIndex << std::endl;
#endif //DISPLAY_DEBUG_FRAMES
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = m_oImgSize.width*y + x;
			const int ldict_idx = uchar_idx;
			const int ushrt_idx = uchar_idx*2;
			const int flt32_idx = uchar_idx*4;
			const uchar nCurrColor = oInputImg.data[uchar_idx];
			int nMinDescDist=s_nDescMaxDataRange_1ch;
			int nMinSumDist=s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+flt32_idx));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
			const int nCurrColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_nLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			int /*nGoodWordsCount=0,*/ nWordIdx=0;
			LocalWord_1ch* pBestLocalWord = NULL;
			float fBestLocalWordDistWeightRatio = FLT_MAX;
			while(nWordIdx<m_nLocalWords) {
				CV_DbgAssert(dynamic_cast<LocalWord_1ch*>(m_aapLocalWords[ldict_idx][nWordIdx]));
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalWords[ldict_idx][nWordIdx];
#if USE_SAMPLES_DEBUG_STRUCT
				const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[uchar_idx];
#else //!USE_SAMPLES_DEBUG_STRUCT
				const uchar& nBGColor = pCurrLocalWord->nColor;
#endif //!USE_SAMPLES_DEBUG_STRUCT
				{
					const int nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
#if USE_SAMPLES_DEBUG_STRUCT
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+ushrt_idx));
#else //!USE_SAMPLES_DEBUG_STRUCT
					const ushort& nBGIntraDesc = pCurrLocalWord->nDesc;
#endif //!USE_SAMPLES_DEBUG_STRUCT
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_nLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					const int nDescDist = (hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc)+hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc))/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					/*if(nSumDist<nMinSumDist) {
						nMinSumDist = nSumDist;
						nMinDescDist = nDescDist;
					}*/
					if(nSumDist<=nCurrColorDistThreshold) {
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						++pCurrLocalWord->nOccurrences;
						float fCurrLocalWordDistWeightRatio = (float)nSumDist/pCurrLocalWord->weight(m_nFrameIndex);
						if(fCurrLocalWordDistWeightRatio<fBestLocalWordDistWeightRatio) {
							fBestLocalWordDistWeightRatio = fCurrLocalWordDistWeightRatio;
							pBestLocalWord = pCurrLocalWord;
							nMinSumDist = nSumDist;
							nMinDescDist = nDescDist;
						}
						if(fCurrLocalWordDistWeightRatio<=nCurrColorDistThreshold/HIGH_WORD_WEIGHT)
							break;
						//if(fCurrLocalWordDistWeightRatio<=nCurrTotColorDistThreshold/AVERAGE_WORD_WEIGHT)
						//	++nGoodWords;
						if((rand()%nLearningRate)==0) { // @@@@@@ should be dictated by T and affected by learning rate override
							pCurrLocalWord->nColor = nCurrColor;
							pCurrLocalWord->nDesc = nCurrIntraDesc;
						}
					}
				}
				failedcheck1ch:
				++nWordIdx;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+ushrt_idx));
			uchar& nLastColor = m_oLastColorFrame.data[uchar_idx];
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(absdiff_uchar(nLastColor,nCurrColor))/s_nColorMaxDataRange_1ch+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/s_nDescMaxDataRange_1ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			float fBestLocalWordWeight = pBestLocalWord?(pBestLocalWord->weight(m_nFrameIndex)):0.0f;
			*pfCurrWeightThreshold = fBestLocalWordWeight; // @@@@ dbg
			if(pBestLocalWord) {
				if(pBestLocalWord->weight(m_nFrameIndex)>AVERAGE_WORD_WEIGHT/(1.0f+*pfCurrDistThresholdVariationFactor)) {
					// == background
					if(fBestLocalWordWeight>HIGH_WORD_WEIGHT/(1.0f+*pfCurrDistThresholdVariationFactor)) {
						// update matching gword in global dictionary
					}
				}
				else {
					// == foreground
					oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				}
			}
			else {
				if(/*gword found for this location with good internal weight & spatial weight (weights depend on S and Dlast for ghost/highvar)*/false) {
					// == background
					// pick a random bad lword, and update it with the gword (and good initial weight)
				}
				else {
					// == foreground
					oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
					if((rand()%nLearningRate)==0) {
						// pick a random bad lword, and replace it with a new one base on the current observation
						const int nRandomWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
						CV_DbgAssert(dynamic_cast<LocalWord_1ch*>(m_aapLocalWords[ldict_idx][nRandomWordIdx]));
						LocalWord_1ch* pNewLocalWord = (LocalWord_1ch*)m_aapLocalWords[ldict_idx][nRandomWordIdx];
						pNewLocalWord->nColor = nCurrColor;
						pNewLocalWord->nDesc = nCurrIntraDesc;
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
						pNewLocalWord->nOccurrences = 1;
					}
				}
			}
			for(int w=1;w<m_nLocalWords; ++w) {
				if(m_aapLocalWords[ldict_idx][w]->weight(m_nFrameIndex) > m_aapLocalWords[ldict_idx][w-1]->weight(m_nFrameIndex)) {
					std::swap(m_aapLocalWords[ldict_idx][w],m_aapLocalWords[ldict_idx][w-1]);
				}
			}
			if(!oCurrFGMask.data[uchar_idx]) {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1))/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
						*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
				}
#if USE_SAMPLES_DEBUG_STRUCT
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nLocalWords;
					*((ushort*)(m_voBGDescSamples[s_rand].data+ushrt_idx)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[uchar_idx] = nCurrColor;
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				int n_rand = rand();
				const int uchar_randidx = m_oImgSize.width*y_rand + x_rand;
				const int flt32_randidx = uchar_randidx*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+flt32_randidx));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+flt32_randidx));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const int ushrt_randidx = uchar_randidx*2;
					int s_rand = rand()%m_nLocalWords;
					*((ushort*)(m_voBGDescSamples[s_rand].data+ushrt_randidx)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[uchar_randidx] = nCurrColor;
				}
#endif //USE_SAMPLES_DEBUG_STRUCT
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
			}
			if((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else if(((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>0) {
					(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<0)
						(*pfCurrDistThresholdVariationFactor) = 0;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSCBLBSP_R_SCALE) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_LOWER;
			}
			nLastIntraDesc = nCurrIntraDesc;
			nLastColor = nCurrColor;
		}
	}
	else { //m_nImgChannels==3
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = m_oImgSize.width*y + x;
			const int ldict_idx = uchar_idx;
			const int flt32_idx = uchar_idx*4;
			const int uchar_rgb_idx = uchar_idx*3;
			const int ushrt_rgb_idx = uchar_rgb_idx*2;
			const uchar* const anCurrColor = oInputImg.data+uchar_rgb_idx;
			int nMinTotDescDist=s_nDescMaxDataRange_3ch;
			int nMinTotSumDist=s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+flt32_idx));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
			const int nCurrTotColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const uchar anCurrIntraLBSPThresholds[3] = {m_nLBSPThreshold_8bitLUT[anCurrColor[0]],m_nLBSPThreshold_8bitLUT[anCurrColor[1]],m_nLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			int /*nGoodWordsCount=0,*/ nWordIdx=0;
			LocalWord_3ch* pBestLocalWord = NULL;
			float fBestLocalWordDistWeightRatio = FLT_MAX;
			while(nWordIdx<m_nLocalWords) {
				CV_DbgAssert(dynamic_cast<LocalWord_3ch*>(m_aapLocalWords[ldict_idx][nWordIdx]));
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalWords[ldict_idx][nWordIdx];
#if USE_SAMPLES_DEBUG_STRUCT
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+ushrt_rgb_idx);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+uchar_rgb_idx;
#else //!USE_SAMPLES_DEBUG_STRUCT
				const uchar* const anBGColor = pCurrLocalWord->anColor;
				const ushort* const anBGIntraDesc = pCurrLocalWord->anDesc;
#endif //!USE_SAMPLES_DEBUG_STRUCT
				int nTotDescDist = 0;
				int nTotSumDist = 0;
				for(int c=0;c<3; ++c) {
					const int nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_nLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					const int nDescDist = (hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]))/2;
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(nTotDescDist<=nCurrTotDescDistThreshold && nTotSumDist<=nCurrTotColorDistThreshold) {
#if DISPLAY_DEBUG_FRAMES
					if(debug_ldict_idx==ldict_idx)
						std::cout << "lword[" << nWordIdx << "] :"
									<< "    age=" << m_nFrameIndex-m_aapLocalWords[ldict_idx][nWordIdx]->nFirstOcc
									<< "    w=" << m_aapLocalWords[ldict_idx][nWordIdx]->weight(m_nFrameIndex) << " (" << AVERAGE_WORD_WEIGHT/(1.0f+*pfCurrDistThresholdVariationFactor) << ")"
									<< "    Ddist=" << (float)nTotDescDist/s_nDescMaxDataRange_3ch << " (" << (float)nCurrTotDescDistThreshold/s_nDescMaxDataRange_3ch << ")"
									<< "    Sdist=" << (float)nTotSumDist/s_nColorMaxDataRange_3ch << " (" << (float)nCurrTotColorDistThreshold/s_nColorMaxDataRange_3ch << ")"
									<< std::endl;
#endif //DISPLAY_DEBUG_FRAMES
					/*if(nTotSumDist<nMinTotSumDist) {
						nMinTotSumDist = nTotSumDist;
						nMinTotDescDist = nTotDescDist;
					}*/
					if(nTotSumDist<=nCurrTotColorDistThreshold) {
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						++pCurrLocalWord->nOccurrences;
						float fCurrLocalWordDistWeightRatio = (float)nTotSumDist/pCurrLocalWord->weight(m_nFrameIndex);
						if(fCurrLocalWordDistWeightRatio<fBestLocalWordDistWeightRatio) {
							fBestLocalWordDistWeightRatio = fCurrLocalWordDistWeightRatio;
							pBestLocalWord = pCurrLocalWord;
							nMinTotSumDist = nTotSumDist;
							nMinTotDescDist = nTotDescDist;
#if DISPLAY_DEBUG_FRAMES
							if(debug_ldict_idx==ldict_idx)
								best_lword_idx = nWordIdx;
#endif //DISPLAY_DEBUG_FRAMES
						}
						if(fCurrLocalWordDistWeightRatio<=nCurrTotColorDistThreshold/HIGH_WORD_WEIGHT)
							break;
						//if(fCurrLocalWordDistWeightRatio<=nCurrTotColorDistThreshold/AVERAGE_WORD_WEIGHT)
						//	++nGoodWords;
						if((rand()%nLearningRate)==0) { // @@@@@@ should be dictated by T and affected by learning rate override
							for(int c=0; c<3; ++c) {
								pCurrLocalWord->anColor[c] = anCurrColor[c];
								pCurrLocalWord->anDesc[c] = anCurrIntraDesc[c];
							}
						}
					}
				}
				failedcheck3ch:
				++nWordIdx;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+ushrt_rgb_idx));
			uchar* anLastColor = m_oLastColorFrame.data+uchar_rgb_idx;
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(L1dist_uchar(anLastColor,anCurrColor))/s_nColorMaxDataRange_3ch+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/s_nDescMaxDataRange_3ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			float fBestLocalWordWeight = pBestLocalWord?(pBestLocalWord->weight(m_nFrameIndex)):0.0f;
			*pfCurrWeightThreshold = fBestLocalWordWeight; // @@@@ dbg
#if DISPLAY_DEBUG_FRAMES
			if(debug_ldict_idx==ldict_idx) {
				if(pBestLocalWord)
					std::cout << "lword[" << best_lword_idx << "] = best; ";
				else
					std::cout << "no lword match; ";
			}
#endif //DISPLAY_DEBUG_FRAMES
			if(pBestLocalWord) {
				if(fBestLocalWordWeight>AVERAGE_WORD_WEIGHT/(1.0f+*pfCurrDistThresholdVariationFactor)) {
					// == background
#if DISPLAY_DEBUG_FRAMES
					if(debug_ldict_idx==ldict_idx)
						std::cout << "background due to high weight; ";
#endif //DISPLAY_DEBUG_FRAMES
					if(fBestLocalWordWeight>HIGH_WORD_WEIGHT/(1.0f+*pfCurrDistThresholdVariationFactor)) {
						// update matching gword in global dictionary
					}
				}
				else {
					// == foreground
					oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
#if DISPLAY_DEBUG_FRAMES
					if(debug_ldict_idx==ldict_idx)
						std::cout << "foreground due to low weight; ";
#endif //DISPLAY_DEBUG_FRAMES
				}
			}
			else {
				if(/*gword found for this location with good internal weight & spatial weight (weights depend on S and Dlast for ghost/highvar)*/false) {
					// == background
					// pick a random bad lword, and update it with the gword (and good initial weight)
#if DISPLAY_DEBUG_FRAMES
					if(debug_ldict_idx==ldict_idx)
						std::cout << "background due to matching gword; ";
#endif //DISPLAY_DEBUG_FRAMES
				}
				else {
					// == foreground
					oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
#if DISPLAY_DEBUG_FRAMES
					if(debug_ldict_idx==ldict_idx)
						std::cout << "foreground due to bad dist match; ";
#endif //DISPLAY_DEBUG_FRAMES
					if((rand()%nLearningRate)==0) {
						// pick a random bad lword, and replace it with a new one base on the current observation
						const int nRandomWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
						CV_DbgAssert(dynamic_cast<LocalWord_3ch*>(m_aapLocalWords[ldict_idx][nRandomWordIdx]));
						LocalWord_3ch* pNewLocalWord = (LocalWord_3ch*)m_aapLocalWords[ldict_idx][nRandomWordIdx];
#if DISPLAY_DEBUG_FRAMES
						if(debug_ldict_idx==ldict_idx)
							std::cout << "replacing lword[" << nRandomWordIdx << "]; ";
#endif //DISPLAY_DEBUG_FRAMES
						for(int c=0; c<3; ++c) {
							pNewLocalWord->anColor[c] = anCurrColor[c];
							pNewLocalWord->anDesc[c] = anCurrIntraDesc[c];
						}
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
						pNewLocalWord->nOccurrences = 1;
					}
				}
			}
#if DISPLAY_DEBUG_FRAMES
			if(debug_ldict_idx==ldict_idx)
				std::cout << std::endl;
#endif //DISPLAY_DEBUG_FRAMES
			for(int w=1;w<m_nLocalWords; ++w) {
				if(m_aapLocalWords[ldict_idx][w]->weight(m_nFrameIndex) > m_aapLocalWords[ldict_idx][w-1]->weight(m_nFrameIndex)) {
					std::swap(m_aapLocalWords[ldict_idx][w],m_aapLocalWords[ldict_idx][w-1]);
				}
			}
			if(!oCurrFGMask.data[uchar_idx]) {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1))/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
						*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
				}
#if USE_SAMPLES_DEBUG_STRUCT
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nLocalWords;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+ushrt_rgb_idx+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+uchar_rgb_idx+c) = anCurrColor[c];
					}
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				int n_rand = rand();
				const int uchar_randidx = m_oImgSize.width*y_rand + x_rand;
				const int flt32_randidx = uchar_randidx*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+flt32_randidx));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+flt32_randidx));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const int uchar_rgb_randidx = uchar_randidx*3;
					const int ushrt_rgb_randidx = uchar_rgb_randidx*2;
					int s_rand = rand()%m_nLocalWords;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+ushrt_rgb_randidx+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+uchar_rgb_randidx+c) = anCurrColor[c];
					}
				}
#endif //USE_SAMPLES_DEBUG_STRUCT
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
			}
			if((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else if(((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>0) {
					(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<0)
						(*pfCurrDistThresholdVariationFactor) = 0;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSCBLBSP_R_SCALE) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_LOWER;
			}
			for(int c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
		}
	}
#if DISPLAY_DEBUG_FRAMES
	/*cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,cv::Size(320,240));
	cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame.at<float>(dbgpt) << std::endl;*/
	/*cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,cv::Size(320,240));
	cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;*/
	/*cv::Mat oMeanSegmResFrameNormalized; m_oMeanSegmResFrame.copyTo(oMeanSegmResFrameNormalized);
	cv::circle(oMeanSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanSegmResFrameNormalized,oMeanSegmResFrameNormalized,cv::Size(320,240));
	cv::imshow("s(x)",oMeanSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " s(" << dbgpt << ") = " << m_oMeanSegmResFrame.at<float>(dbgpt) << std::endl;*/
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,1.0f/BGSCBLBSP_R_UPPER,-BGSCBLBSP_R_LOWER/BGSCBLBSP_R_UPPER);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,cv::Size(320,240));
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdVariationFrameNormalized; cv::normalize(m_oDistThresholdVariationFrame,oDistThresholdVariationFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(255));
	cv::resize(oDistThresholdVariationFrameNormalized,oDistThresholdVariationFrameNormalized,cv::Size(320,240));
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSCBLBSP_T_UPPER,-BGSCBLBSP_T_LOWER/BGSCBLBSP_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,cv::Size(320,240));
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_DEBUG_FRAMES
	cv::imshow("s(x)",oCurrFGMask);
	cv::imshow("w(x)",m_oWeightThresholdFrame);
	cv::bitwise_xor(oCurrFGMask,m_oPureFGMask_last,m_oPureFGBlinkMask_curr);
	cv::bitwise_or(m_oPureFGBlinkMask_curr,m_oPureFGBlinkMask_last,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oTempFGMask);
	cv::bitwise_and(m_oBlinksFrame,m_oTempFGMask,m_oBlinksFrame);
	m_oPureFGBlinkMask_curr.copyTo(m_oPureFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oPureFGMask_last);
	cv::morphologyEx(oCurrFGMask,m_oTempFGMask,cv::MORPH_CLOSE,cv::Mat());
	cv::floodFill(m_oTempFGMask,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oTempFGMask,m_oTempFGMask);
	cv::bitwise_or(oCurrFGMask,m_oTempFGMask,oCurrFGMask);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat());
	cv::bitwise_not(m_oFGMask_last_dilated,m_oTempFGMask);
	cv::bitwise_and(m_oBlinksFrame,m_oTempFGMask,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
}

void BackgroundSubtractorCBLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC(m_nImgChannels));
	// @@@@@@ TO BE REWRITTEN FOR WORD-BASED RECONSTRUCTION
	/*for(int w=0; w<m_nLocalWords; ++w) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				int img_idx = m_voBGColorSamples[w].step.p[0]*y + m_voBGColorSamples[w].step.p[1]*x;
				int flt32_idx = img_idx*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+flt32_idx);
				uchar* oBGImgPtr = m_voBGColorSamples[w].data+img_idx;
				for(int c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nLocalWords;
			}
		}
	}*/
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

void BackgroundSubtractorCBLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_Assert(LBSP::DESC_SIZE==2);
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC(m_nImgChannels));
	// @@@@@@ TO BE REWRITTEN FOR WORD-BASED RECONSTRUCTION
	/*for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				int desc_idx = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
				int flt32_idx = desc_idx*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+flt32_idx);
				ushort* oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+desc_idx);
				for(int c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
			}
		}
	}*/
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

void BackgroundSubtractorCBLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
}

BackgroundSubtractorCBLBSP::LocalWord::LocalWord(int& nWIDSeed)
	:	 nWID(++nWIDSeed)
		,nFirstOcc(-1)
		,nLastOcc(0)
		,nOccurrences(1)
{}

float BackgroundSubtractorCBLBSP::LocalWord::weight(int nCurrFrame) {
	return (float)(nOccurrences)/((nLastOcc-nFirstOcc)+(nCurrFrame-nLastOcc)+LWORD_WEIGHT_OFFSET);
}

BackgroundSubtractorCBLBSP::GlobalWord::GlobalWord(cv::Size oFrameSize)
	:	 oSpatioOccMap(oFrameSize,CV_8UC1) {}

BackgroundSubtractorCBLBSP::LocalWord_1ch::LocalWord_1ch(int& nWIDSeed)
	:	 LocalWord(nWIDSeed), nColor(0), nDesc(0) {}

float BackgroundSubtractorCBLBSP::LocalWord_1ch::distance(LocalWord* w) {
	CV_DbgAssert(dynamic_cast<LocalWord_1ch*>(w));
	return ((float)(absdiff_uchar(nColor,((LocalWord_1ch*)w)->nColor))/s_nColorMaxDataRange_1ch + (float)(hdist_ushort_8bitLUT(nDesc,((LocalWord_1ch*)w)->nDesc))/s_nDescMaxDataRange_1ch)/2;
}

BackgroundSubtractorCBLBSP::LocalWord_3ch::LocalWord_3ch(int& nWIDSeed)
	:	 LocalWord(nWIDSeed), anColor({0,0,0}), anDesc({0,0,0}) {}

float BackgroundSubtractorCBLBSP::LocalWord_3ch::distance(LocalWord* w) {
	CV_DbgAssert(dynamic_cast<LocalWord_3ch*>(w));
	return ((float)(L1dist_uchar(anColor,((LocalWord_3ch*)w)->anColor))/s_nColorMaxDataRange_3ch+(float)(hdist_ushort_8bitLUT(anDesc,((LocalWord_3ch*)w)->anDesc))/s_nDescMaxDataRange_3ch)/2;
}

BackgroundSubtractorCBLBSP::GlobalWord_1ch::GlobalWord_1ch(int& nWIDSeed, cv::Size oFrameSize)
	:	 LocalWord_1ch(nWIDSeed), GlobalWord(oFrameSize) {}

float BackgroundSubtractorCBLBSP::GlobalWord_1ch::distance(LocalWord*) {
	CV_Error(1,"@@@@");
	return 0; //@@@@@
}

BackgroundSubtractorCBLBSP::GlobalWord_3ch::GlobalWord_3ch(int& nWIDSeed, cv::Size oFrameSize)
	:	 LocalWord_3ch(nWIDSeed), GlobalWord(oFrameSize) {}

float BackgroundSubtractorCBLBSP::GlobalWord_3ch::distance(LocalWord*) {
	CV_Error(1,"@@@@");
	return 0; //@@@@@
}

