#include "BackgroundSubtractorCBLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// local define used for debug purposes only
#define DISPLAY_CBLBSP_DEBUG_INFO 0
// local define used to specify the bootstrap window size
#define BOOTSTRAP_WIN_SIZE 250
// local define used to specify the persistence of a local illumination update indicator
#define ILLUMUPDT_REGION_DEFAULT_VAL 20
// local define used to specify whether to use single channel analysis or not (for RGB images only)
#define USE_SC_THRS_VALIDATION 1
// local define used to specify whether to use hard desc dist threshold checks or not
#define USE_HARD_SC_DESC_DIST_CHECKS 0
// local define for the base nb of local words
#define LWORD_BASE_COUNT 3
// local define for the base nb of global words
#define GWORD_BASE_COUNT 0
// local define for the gword update rate
#define GWORD_UPDATE_RATE 8
// local define for the gword decimation factor
#define GWORD_WEIGHT_DECIMATION_FACTOR 0.1f
// local define for the amount of weight offset to apply to words, making sure new words aren't always better than old ones
#define LWORD_WEIGHT_OFFSET 1000
// local define for the initial weight of a new word (used to make sure old words aren't worse off than new seeds)
#define LWORD_INIT_WEIGHT (1.0f/LWORD_WEIGHT_OFFSET)
// local define for the maximum weight a word can achieve before cutting off occ incr (used to make sure model stays good for long-term uses)
#define LWORD_MAX_WEIGHT 5.0f
// local define used to specify the debug window size
#define DEBUG_WINDOW_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define UNSTAB_COLOR_DIST_OFFSET 5
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET 2
// local define used to activate internal HRC's to time different algorithm sections
#define USE_INTERNAL_HRCS 0
// local define used to activate memory checks throughout different algorithm sections
#define USE_INTERNAL_RCHECKS 0
// local define used to determine at what continuous final FG-to-BG ratio to reset the model
#define MODEL_RESET_MIN_FINAL_FG_RATIO 0.75f
// local define used to determine how long the min ratio must be kept for a model reset
#define MODEL_RESET_MIN_FRAME_COUNT 10

#if USE_INTERNAL_HRCS
#include "PlatformUtils.h"
#if !PLATFORM_SUPPORTS_CPP11
#error "Cannnot activate HRCs, C++11 not supported."
#endif //!PLATFORM_SUPPORTS_CPP11
#endif //USE_INTERNAL_HRCS
#if USE_INTERNAL_RCHECKS
#include "PlatformUtils.h"
#if WIN32 || PLATFORM_USES_WIN32API
#error "Windows internal memory checks not implemented."
#endif //WIN32 || PLATFORM_USES_WIN32API
#include <unistd.h>
#include <sys/resource.h>
#include <stdio.h>
inline size_t getCurrentRSS() {
	long rss = 0L;
	FILE* fp = NULL;
	if((fp=fopen("/proc/self/statm","r"))==NULL)
		return (size_t)0L;
	if(fscanf(fp,"%*s%ld",&rss)!=1) {
		fclose( fp );
		return (size_t)0L;
	}
	fclose(fp);
	return (size_t)rss*(size_t)sysconf(_SC_PAGESIZE);
}
#endif //USE_INTERNAL_RCHECKS

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorCBLBSP::BackgroundSubtractorCBLBSP(	 float fLBSPThreshold
														,size_t nLBSPThresholdOffset
														,size_t nInitDescDistThreshold
														,size_t nInitColorDistThreshold
														,float fLocalWordsPerChannel
														,float fGlobalWordsPerChannel)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold,nLBSPThresholdOffset)
		,m_bInitializedInternalStructs(false)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_fLocalWordsPerChannel(fLocalWordsPerChannel)
		,m_nLocalWords(0)
		,m_fGlobalWordsPerChannel(fGlobalWordsPerChannel)
		,m_nGlobalWords(0)
		,m_nMaxLocalDictionaries(0)
		,m_nFrameIndex(SIZE_MAX)
		,m_nModelResetFrameCount(0)
		,m_aapLocalDicts(nullptr)
		,m_apLocalWordList_1ch(nullptr)
		,m_apLocalWordListIter_1ch(nullptr)
		,m_apLocalWordList_3ch(nullptr)
		,m_apLocalWordListIter_3ch(nullptr)
		,m_apGlobalDict(nullptr)
		,m_apGlobalWordList_1ch(nullptr)
		,m_apGlobalWordListIter_1ch(nullptr)
		,m_apGlobalWordList_3ch(nullptr)
		,m_apGlobalWordListIter_3ch(nullptr)
		,m_apGlobalWordLookupTable_BG(nullptr)
		,m_apGlobalWordLookupTable_FG(nullptr) {
	CV_Assert(m_fLocalWordsPerChannel>=1.0f && m_fGlobalWordsPerChannel>=1.0f);
}

BackgroundSubtractorCBLBSP::~BackgroundSubtractorCBLBSP() {
	CleanupDictionaries();
}

void BackgroundSubtractorCBLBSP::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	// == init
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
	CleanupDictionaries();
	m_voKeyPoints = voNewKeyPoints;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nMaxLocalDictionaries = oInitImg.cols*oInitImg.rows;
	m_nLocalWords = ((size_t)(m_fLocalWordsPerChannel*m_nImgChannels)) + LWORD_BASE_COUNT;
	m_nGlobalWords = ((size_t)(m_fGlobalWordsPerChannel*m_nImgChannels)) + GWORD_BASE_COUNT;
	m_nFrameIndex = 0;
	m_nModelResetFrameCount = 0;
	m_aapLocalDicts = new LocalWord*[m_nMaxLocalDictionaries*m_nLocalWords];
	memset(m_aapLocalDicts,0,sizeof(LocalWord*)*m_nMaxLocalDictionaries*m_nLocalWords);
	m_apGlobalDict = new GlobalWord*[m_nGlobalWords];
	memset(m_apGlobalDict,0,sizeof(GlobalWord*)*m_nGlobalWords);
	m_apGlobalWordLookupTable_BG = new GlobalWord*[m_nMaxLocalDictionaries];
	memset(m_apGlobalWordLookupTable_BG,0,sizeof(GlobalWord*)*m_nMaxLocalDictionaries);
	m_apGlobalWordLookupTable_FG = new GlobalWord*[m_nMaxLocalDictionaries];
	memset(m_apGlobalWordLookupTable_FG,0,sizeof(GlobalWord*)*m_nMaxLocalDictionaries);
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(BGSCBLBSP_R_LOWER);
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(10.0f);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(BGSCBLBSP_T_LOWER);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_burst.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_burst = cv::Scalar(0.0f);
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanLastDistFrame_burst.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame_burst = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_burst.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_burst = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame = cv::Scalar(0.0f);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
	m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
	m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
	m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_last = cv::Scalar_<uchar>(0);
	m_oTempGlobalWordWeightDiffFactor.create(m_oImgSize,CV_32FC1);
	m_oTempGlobalWordWeightDiffFactor = cv::Scalar(-GWORD_WEIGHT_DECIMATION_FACTOR);
	m_oPureFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last2.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last2 = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated_inverted = cv::Scalar_<uchar>(0);
	m_oHighVarRegionMask.create(m_oImgSize,CV_8UC1);
	m_oHighVarRegionMask = cv::Scalar_<uchar>(0);
	m_oGhostRegionMask.create(m_oImgSize,CV_8UC1);
	m_oGhostRegionMask = cv::Scalar_<uchar>(1);
	m_oUnstableRegionMask.create(m_oImgSize,CV_8UC1);
	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
	m_oIllumUpdtRegionMask.create(m_oImgSize,CV_8UC1);
	m_oIllumUpdtRegionMask = cv::Scalar_<uchar>(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT+m_nAbsLBSPThreshold);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_color = idx_uchar;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
		m_apLocalWordList_1ch = new LocalWord_1ch[nKeyPoints*m_nLocalWords];
		memset(m_apLocalWordList_1ch,0,sizeof(LocalWord_1ch)*nKeyPoints*m_nLocalWords);
		m_apLocalWordListIter_1ch = m_apLocalWordList_1ch;
		m_apGlobalWordList_1ch = new GlobalWord_1ch[m_nGlobalWords];
		m_apGlobalWordListIter_1ch = m_apGlobalWordList_1ch;
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nAbsLBSPThreshold);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_color = idx_uchar*3;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			for(size_t c=0; c<3; ++c) {
				m_oLastColorFrame.data[idx_color+c] = oInitImg.data[idx_color+c];
				LBSP::computeSingleRGBDescriptor(oInitImg,oInitImg.data[idx_color+c],x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color+c]],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
		m_apLocalWordList_3ch = new LocalWord_3ch[nKeyPoints*m_nLocalWords];
		memset(m_apLocalWordList_3ch,0,sizeof(LocalWord_3ch)*nKeyPoints*m_nLocalWords);
		m_apLocalWordListIter_3ch = m_apLocalWordList_3ch;
		m_apGlobalWordList_3ch = new GlobalWord_3ch[m_nGlobalWords];
		m_apGlobalWordListIter_3ch = m_apGlobalWordList_3ch;
	}
	m_bInitializedInternalStructs = true;
	refreshModel(1,1,0.0f);
	m_bInitialized = true;
}

void BackgroundSubtractorCBLBSP::refreshModel(size_t nBaseOccCount, size_t nOverallMatchOccIncr, float fOccDecrFrac, bool bForceFGUpdate) {
	// == refresh
	CV_Assert(m_bInitializedInternalStructs);
	CV_Assert(fOccDecrFrac>=0.0f && fOccDecrFrac<=1.0f);
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_orig_uchar])
				continue;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_flt32 = idx_orig_uchar*4;
			const float fCurrDistThresholdFactor = *(float*)(m_oDistThresholdFrame.data+idx_orig_flt32);
			const size_t nCurrColorDistThreshold = (size_t)((fCurrDistThresholdFactor*m_nColorDistThreshold-((!m_oUnstableRegionMask.data[idx_orig_uchar])*UNSTAB_COLOR_DIST_OFFSET))*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(fCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_orig_uchar]*UNSTAB_DESC_DIST_OFFSET);
			if(fOccDecrFrac>0.0f) {
				for(size_t nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_1ch* pCurrLocalWord = ((LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]);
					if(pCurrLocalWord)
						pCurrLocalWord->nOccurrences -= (size_t)(fOccDecrFrac*pCurrLocalWord->nOccurrences);
				}
			}
			const size_t nLocalIters = (s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight)*2;
			const size_t nLocalWordOccIncr = std::max(nOverallMatchOccIncr/nLocalIters,(size_t)1);
			for(size_t n=0; n<nLocalIters; ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_uchar = m_oImgSize.width*y_sample + x_sample;
				if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_sample_uchar])
					continue;
				const size_t idx_sample_color = idx_sample_uchar;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar nSampleColor = m_oLastColorFrame.data[idx_sample_color];
				const ushort nSampleIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_1ch* pCurrLocalWord = ((LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]);
					if(pCurrLocalWord
							&& absdiff_uchar(nSampleColor,pCurrLocalWord->nColor)<=nCurrColorDistThreshold
							&& hdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc)<=nCurrDescDistThreshold) {
						pCurrLocalWord->nOccurrences += nLocalWordOccIncr;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						break;
					}
				}
				if(nLocalWordIdx==m_nLocalWords) {
					nLocalWordIdx = m_nLocalWords-1;
					LocalWord_1ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx])
						pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					else {
						pCurrLocalWord = m_apLocalWordListIter_1ch++;
						m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
					}
					pCurrLocalWord->nColor = nSampleColor;
					pCurrLocalWord->nDesc = nSampleIntraDesc;
					pCurrLocalWord->nOccurrences = nBaseOccCount;
					pCurrLocalWord->nFirstOcc = m_nFrameIndex;
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
				}
				while(nLocalWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1] || GetLocalWordWeight(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1],m_nFrameIndex))) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]);
					--nLocalWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nLocalWordIdx=1; nLocalWordIdx<m_nLocalWords; ++nLocalWordIdx) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = m_apLocalWordListIter_1ch++;
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(nCurrColorDistThreshold+1))-(int)nCurrColorDistThreshold/2;
					pCurrLocalWord->nColor = cv::saturate_cast<uchar>((int)pRefLocalWord->nColor+nRandColorOffset);
					pCurrLocalWord->nDesc = pRefLocalWord->nDesc;
					pCurrLocalWord->nOccurrences = (size_t)(pRefLocalWord->nOccurrences*((float)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords));
					pCurrLocalWord->nFirstOcc = m_nFrameIndex;
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_1ch==(m_apLocalWordListIter_1ch-nKeyPoints*m_nLocalWords));
		cv::Mat oGlobalDictPresenceLookupMap(m_oImgSize,CV_8UC1);
		oGlobalDictPresenceLookupMap = cv::Scalar_<uchar>(0);
		size_t nLocalDictIterIncr = (nKeyPoints/m_nGlobalWords)>0?(nKeyPoints/m_nGlobalWords):1;
		for(size_t k=0; k<nKeyPoints; k+=nLocalDictIterIncr) { // <=(m_nGlobalWords) gwords from (m_nGlobalWords) equally spaced keypoints
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_orig_uchar])
				continue;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_flt32 = idx_orig_uchar*4;
			const float fCurrDistThresholdFactor = *(float*)(m_oDistThresholdFrame.data+idx_orig_flt32);
			const size_t nCurrColorDistThreshold = (size_t)((fCurrDistThresholdFactor*m_nColorDistThreshold-((!m_oUnstableRegionMask.data[idx_orig_uchar])*UNSTAB_COLOR_DIST_OFFSET))*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(fCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_orig_uchar]*UNSTAB_DESC_DIST_OFFSET);
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			const LocalWord_1ch* pRefBestLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict];
			const float fRefBestLocalWordWeight = GetLocalWordWeight(pRefBestLocalWord,m_nFrameIndex);
			const uchar nRefBestLocalWordDescBITS = popcount_ushort_8bitsLUT(pRefBestLocalWord->nDesc);
			size_t nGlobalWordIdx; GlobalWord_1ch* pCurrGlobalWord;
			for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
				pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
				if(pCurrGlobalWord
						&& absdiff_uchar(pCurrGlobalWord->nColor,pRefBestLocalWord->nColor)<=nCurrColorDistThreshold
						&& absdiff_uchar(nRefBestLocalWordDescBITS,pCurrGlobalWord->nDescBITS)<=nCurrDescDistThreshold/2)
					break;
			}
			if(nGlobalWordIdx==m_nGlobalWords) {
				nGlobalWordIdx = m_nGlobalWords-1;
				if(m_apGlobalDict[nGlobalWordIdx])
					pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
				else {
					pCurrGlobalWord = m_apGlobalWordListIter_1ch++;
					m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				}
				pCurrGlobalWord->nColor = pRefBestLocalWord->nColor;
				pCurrGlobalWord->nDescBITS = nRefBestLocalWordDescBITS;
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
			}
			m_apGlobalWordLookupTable_BG[idx_orig_uchar] = pCurrGlobalWord;
			float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_flt32);
			if((*pfCurrGlobalWordLocalWeight)<fRefBestLocalWordWeight) {
				pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
				(*pfCurrGlobalWordLocalWeight) += fRefBestLocalWordWeight;
			}
			oGlobalDictPresenceLookupMap.data[idx_orig_uchar] = UCHAR_MAX;
			while(nGlobalWordIdx>0 && (!m_apGlobalDict[nGlobalWordIdx-1] || m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)) {
				std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
				--nGlobalWordIdx;
			}
		}
		size_t nLocalDictWordIdxOffset = 0;
		size_t nLookupMapIdxOffset = (nLocalDictIterIncr/2>0)?(nLocalDictIterIncr/2):1;
		while((size_t)(m_apGlobalWordListIter_1ch-m_apGlobalWordList_1ch)<m_nGlobalWords) {
			if(nLocalDictWordIdxOffset<m_nLocalWords) {
				size_t nLookupMapIdx = 0;
				const size_t nColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
				const size_t nDescDistThreshold = 2+m_nDescDistThreshold;
				while(nLookupMapIdx<nKeyPoints && (size_t)(m_apGlobalWordListIter_1ch-m_apGlobalWordList_1ch)<m_nGlobalWords) {
					if(m_aapLocalDicts[nLookupMapIdx*m_nLocalWords] && oGlobalDictPresenceLookupMap.data[nLookupMapIdx]<UCHAR_MAX && (!m_oFGMask_last_dilated.data[nLookupMapIdx]||bForceFGUpdate)) {
						const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[nLookupMapIdx*m_nLocalWords+nLocalDictWordIdxOffset];
						const float fRefLocalWordWeight = GetLocalWordWeight(pRefLocalWord,m_nFrameIndex);
						const uchar nRefLocalWordDescBITS = popcount_ushort_8bitsLUT(pRefLocalWord->nDesc);
						size_t nGlobalWordIdx; GlobalWord_1ch* pCurrGlobalWord;
						for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
							pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							if(pCurrGlobalWord
									&& absdiff_uchar(pCurrGlobalWord->nColor,pRefLocalWord->nColor)<=nColorDistThreshold
									&& absdiff_uchar(nRefLocalWordDescBITS,pCurrGlobalWord->nDescBITS)<=nDescDistThreshold/2)
								break;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							if(m_apGlobalDict[nGlobalWordIdx])
								pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							else {
								pCurrGlobalWord = m_apGlobalWordListIter_1ch++;
								m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							}
							pCurrGlobalWord->nColor = pRefLocalWord->nColor;
							pCurrGlobalWord->nDescBITS = nRefLocalWordDescBITS;
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
						}
						m_apGlobalWordLookupTable_BG[nLookupMapIdx] = pCurrGlobalWord;
						float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4));
						if((*pfCurrGlobalWordLocalWeight)<fRefLocalWordWeight) {
							pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
							(*pfCurrGlobalWordLocalWeight) += fRefLocalWordWeight;
						}
						oGlobalDictPresenceLookupMap.data[nLookupMapIdx] = UCHAR_MAX;
						while(nGlobalWordIdx>0 && (!m_apGlobalDict[nGlobalWordIdx-1] || m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)) {
							std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
							--nGlobalWordIdx;
						}
					}
					nLookupMapIdx += nLookupMapIdxOffset;
				}
				nLookupMapIdxOffset = (nLookupMapIdxOffset/2>0)?(nLookupMapIdxOffset/2):1;
				++nLocalDictWordIdxOffset;
			}
			else {
				size_t nGlobalWordFillIdx = (size_t)(m_apGlobalWordListIter_1ch-m_apGlobalWordList_1ch);
				while(nGlobalWordFillIdx<m_nGlobalWords) {
					CV_Assert(!m_apGlobalDict[nGlobalWordFillIdx]);
					GlobalWord_1ch* pCurrGlobalWord = m_apGlobalWordListIter_1ch++;
					pCurrGlobalWord->nColor = rand()%(UCHAR_MAX+1);
					pCurrGlobalWord->nDescBITS = 0;
					pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
					pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
					pCurrGlobalWord->fLatestWeight = 0.0f;
					m_apGlobalDict[nGlobalWordFillIdx++] = pCurrGlobalWord;
				}
				break;
			}
		}
		CV_Assert((size_t)(m_apGlobalWordListIter_1ch-m_apGlobalWordList_1ch)==m_nGlobalWords && m_apGlobalWordList_1ch==(m_apGlobalWordListIter_1ch-m_nGlobalWords));
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_orig_uchar])
				continue;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_flt32 = idx_orig_uchar*4;
			const float fCurrDistThresholdFactor = *(float*)(m_oDistThresholdFrame.data+idx_orig_flt32);
			const size_t nCurrTotColorDistThreshold = (size_t)((fCurrDistThresholdFactor*m_nColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_orig_uchar])*UNSTAB_COLOR_DIST_OFFSET))*3;
			const size_t nCurrTotDescDistThreshold = (((size_t)1<<((size_t)floor(fCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_orig_uchar]*UNSTAB_DESC_DIST_OFFSET))*3;
			if(fOccDecrFrac>0.0f) {
				for(size_t nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_3ch* pCurrLocalWord = ((LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]);
					if(pCurrLocalWord)
						pCurrLocalWord->nOccurrences -= (size_t)(fOccDecrFrac*pCurrLocalWord->nOccurrences);
				}
			}
			const size_t nLocalIters = (s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight)*2;
			const size_t nLocalWordOccIncr = std::max(nOverallMatchOccIncr/nLocalIters,(size_t)1);
			for(size_t n=0; n<nLocalIters; ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_uchar = m_oImgSize.width*y_sample + x_sample;
				if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_sample_uchar])
					continue;
				const size_t idx_sample_color = idx_sample_uchar*3;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar* const anSampleColor = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const anSampleIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					if(pCurrLocalWord
							&& L1dist_uchar(anSampleColor,pCurrLocalWord->anColor)<=nCurrTotColorDistThreshold
							&& hdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc)<=nCurrTotDescDistThreshold) {
						pCurrLocalWord->nOccurrences += nLocalWordOccIncr;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						break;
					}
				}
				if(nLocalWordIdx==m_nLocalWords) {
					nLocalWordIdx = m_nLocalWords-1;
					LocalWord_3ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx])
						pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					else {
						pCurrLocalWord = m_apLocalWordListIter_3ch++;
						m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
					}
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = anSampleColor[c];
						pCurrLocalWord->anDesc[c] = anSampleIntraDesc[c];
					}
					pCurrLocalWord->nOccurrences = nBaseOccCount;
					pCurrLocalWord->nFirstOcc = m_nFrameIndex;
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
				}
				while(nLocalWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1] || GetLocalWordWeight(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1],m_nFrameIndex))) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]);
					--nLocalWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nLocalWordIdx=1; nLocalWordIdx<m_nLocalWords; ++nLocalWordIdx) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = m_apLocalWordListIter_3ch++;
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(nCurrTotColorDistThreshold/3+1))-(int)(nCurrTotColorDistThreshold/6);
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = cv::saturate_cast<uchar>((int)pRefLocalWord->anColor[c]+nRandColorOffset);
						pCurrLocalWord->anDesc[c] = pRefLocalWord->anDesc[c];
					}
					pCurrLocalWord->nOccurrences = (size_t)(pRefLocalWord->nOccurrences*((float)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords));
					pCurrLocalWord->nFirstOcc = m_nFrameIndex;
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_3ch==(m_apLocalWordListIter_3ch-nKeyPoints*m_nLocalWords));
		cv::Mat oGlobalDictPresenceLookupMap(m_oImgSize,CV_8UC1);
		oGlobalDictPresenceLookupMap = cv::Scalar_<uchar>(0);
		size_t nLocalDictIterIncr = (nKeyPoints/m_nGlobalWords)>0?(nKeyPoints/m_nGlobalWords):1;
		for(size_t k=0; k<nKeyPoints; k+=nLocalDictIterIncr) { // <=(m_nGlobalWords) gwords from (m_nGlobalWords) equally spaced keypoints
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			if(!bForceFGUpdate && m_oFGMask_last_dilated.data[idx_orig_uchar])
				continue;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_flt32 = idx_orig_uchar*4;
			const float fCurrDistThresholdFactor = *(float*)(m_oDistThresholdFrame.data+idx_orig_flt32);
			const size_t nCurrTotColorDistThreshold = (size_t)((fCurrDistThresholdFactor*m_nColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_orig_uchar])*UNSTAB_COLOR_DIST_OFFSET))*3;
			const size_t nCurrTotDescDistThreshold = (((size_t)1<<((size_t)floor(fCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_orig_uchar]+UNSTAB_DESC_DIST_OFFSET))*3;
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			const LocalWord_3ch* pRefBestLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict];
			const float fRefBestLocalWordWeight = GetLocalWordWeight(pRefBestLocalWord,m_nFrameIndex);
			const uchar nRefBestLocalWordDescBITS = popcount_ushort_8bitsLUT(pRefBestLocalWord->anDesc);
			size_t nGlobalWordIdx; GlobalWord_3ch* pCurrGlobalWord;
			for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
				pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
				if(pCurrGlobalWord
						&& (L1dist_uchar(pRefBestLocalWord->anColor,pCurrGlobalWord->anColor)/2+cdist_uchar(pRefBestLocalWord->anColor,pCurrGlobalWord->anColor)*6)<=nCurrTotColorDistThreshold
						&& absdiff_uchar(nRefBestLocalWordDescBITS,pCurrGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold/2)
					break;
			}
			if(nGlobalWordIdx==m_nGlobalWords) {
				nGlobalWordIdx = m_nGlobalWords-1;
				if(m_apGlobalDict[nGlobalWordIdx])
					pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
				else {
					pCurrGlobalWord = m_apGlobalWordListIter_3ch++;
					m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				}
				for(size_t c=0; c<3; ++c)
					pCurrGlobalWord->anColor[c] = pRefBestLocalWord->anColor[c];
				pCurrGlobalWord->nDescBITS = nRefBestLocalWordDescBITS;
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
			}
			m_apGlobalWordLookupTable_BG[idx_orig_uchar] = pCurrGlobalWord;
			float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_flt32);
			if((*pfCurrGlobalWordLocalWeight)<fRefBestLocalWordWeight) {
				pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
				(*pfCurrGlobalWordLocalWeight) += fRefBestLocalWordWeight;
			}
			oGlobalDictPresenceLookupMap.data[idx_orig_uchar] = UCHAR_MAX;
			while(nGlobalWordIdx>0 && (!m_apGlobalDict[nGlobalWordIdx-1] || m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)) {
				std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
				--nGlobalWordIdx;
			}
		}
		size_t nLocalDictWordIdxOffset = 0;
		size_t nLookupMapIdxOffset = (nLocalDictIterIncr/2>0)?(nLocalDictIterIncr/2):1;
		while((size_t)(m_apGlobalWordListIter_3ch-m_apGlobalWordList_3ch)<m_nGlobalWords) {
			if(nLocalDictWordIdxOffset<m_nLocalWords) {
				size_t nLookupMapIdx = 0;
				const size_t nTotColorDistThreshold = m_nColorDistThreshold*3;
				const size_t nTotDescDistThreshold = (2+m_nDescDistThreshold)*3;
				while(nLookupMapIdx<nKeyPoints && (size_t)(m_apGlobalWordListIter_3ch-m_apGlobalWordList_3ch)<m_nGlobalWords) {
					if(m_aapLocalDicts[nLookupMapIdx*m_nLocalWords] && oGlobalDictPresenceLookupMap.data[nLookupMapIdx]<UCHAR_MAX && (!m_oFGMask_last_dilated.data[nLookupMapIdx]||bForceFGUpdate)) {
						const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[nLookupMapIdx*m_nLocalWords+nLocalDictWordIdxOffset];
						const float fRefLocalWordWeight = GetLocalWordWeight(pRefLocalWord,m_nFrameIndex);
						const uchar nRefLocalWordDescBITS = popcount_ushort_8bitsLUT(pRefLocalWord->anDesc);
						size_t nGlobalWordIdx; GlobalWord_3ch* pCurrGlobalWord;
						for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
							pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							if(pCurrGlobalWord
									&& (L1dist_uchar(pRefLocalWord->anColor,pCurrGlobalWord->anColor)/2+cdist_uchar(pRefLocalWord->anColor,pCurrGlobalWord->anColor)*6)<=nTotColorDistThreshold
									&& absdiff_uchar(nRefLocalWordDescBITS,pCurrGlobalWord->nDescBITS)<=nTotDescDistThreshold)
								break;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							if(m_apGlobalDict[nGlobalWordIdx])
								pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							else {
								pCurrGlobalWord = m_apGlobalWordListIter_3ch++;
								m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							}
							for(size_t c=0; c<3; ++c)
								pCurrGlobalWord->anColor[c] = pRefLocalWord->anColor[c];
							pCurrGlobalWord->nDescBITS = nRefLocalWordDescBITS;
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
						}
						m_apGlobalWordLookupTable_BG[nLookupMapIdx] = pCurrGlobalWord;
						float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4));
						if((*pfCurrGlobalWordLocalWeight)<fRefLocalWordWeight) {
							pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
							(*pfCurrGlobalWordLocalWeight) += fRefLocalWordWeight;
						}
						oGlobalDictPresenceLookupMap.data[nLookupMapIdx] = UCHAR_MAX;
						while(nGlobalWordIdx>0 && (!m_apGlobalDict[nGlobalWordIdx-1] || m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)) {
							std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
							--nGlobalWordIdx;
						}
					}
					nLookupMapIdx += nLookupMapIdxOffset;
				}
				nLookupMapIdxOffset = (nLookupMapIdxOffset/2>0)?(nLookupMapIdxOffset/2):1;
				++nLocalDictWordIdxOffset;
			}
			else {
				size_t nGlobalWordFillIdx = (size_t)(m_apGlobalWordListIter_3ch-m_apGlobalWordList_3ch);
				while(nGlobalWordFillIdx<m_nGlobalWords) {
					CV_Assert(!m_apGlobalDict[nGlobalWordFillIdx]);
					GlobalWord_3ch* pCurrGlobalWord = m_apGlobalWordListIter_3ch++;
					for(size_t c=0; c<3; ++c)
						pCurrGlobalWord->anColor[c] = rand()%(UCHAR_MAX+1);
					pCurrGlobalWord->nDescBITS = 0;
					pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
					pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
					pCurrGlobalWord->fLatestWeight = 0.0f;
					m_apGlobalDict[nGlobalWordFillIdx++] = pCurrGlobalWord;
				}
				break;
			}
		}
		CV_Assert((size_t)(m_apGlobalWordListIter_3ch-m_apGlobalWordList_3ch)==m_nGlobalWords && m_apGlobalWordList_3ch==(m_apGlobalWordListIter_3ch-m_nGlobalWords));
	}
}

void BackgroundSubtractorCBLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	// == process
#if USE_INTERNAL_RCHECKS
	size_t start_rss = getCurrentRSS();
#endif //USE_INTERNAL_RCHECKS
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	const size_t nKeyPoints = m_voKeyPoints.size();
	const size_t nDefaultWordOccIncr = m_nFrameIndex<BOOTSTRAP_WIN_SIZE?8:m_nFrameIndex<BOOTSTRAP_WIN_SIZE*2?4:1;
	const float fRollAvgFactor = 1.0f/std::min(++m_nFrameIndex,(size_t)BGSCBLBSP_N_SAMPLES_FOR_MEAN);
	const float fRollAvgFactor_burst = 1.0f/std::min(m_nFrameIndex,(size_t)BGSCBLBSP_N_SAMPLES_FOR_MEAN/4);
	const size_t nCurrGlobalWordUpdateRate = GWORD_UPDATE_RATE;
#if DISPLAY_CBLBSP_DEBUG_INFO
	std::vector<std::string> vsWordModList(m_nMaxLocalDictionaries*m_nLocalWords);
	uchar anDBGColor[3] = {0,0,0};
	ushort anDBGIntraDesc[3] = {0,0,0};
	bool bDBGMaskResult = false;
	bool bDBGMaskModifiedByGDict = false;
	GlobalWord* pDBGGlobalWordModifier = nullptr;
	float fDBGGlobalWordModifierLocalWeight = 0.0f;
	float fDBGLocalWordsWeightSumThreshold = 0.0f;
	size_t idx_dbg_ldict = UINT_MAX;
	size_t nDBGWordOccIncr = nDefaultWordOccIncr;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
#if USE_INTERNAL_HRCS
	float fPrepTimeSum_MS = 0.0f;
	float fLDictScanTimeSum_MS = 0.0f;
	float fBGRawTimeSum_MS = 0.0f;
	float fFGRawTimeSum_MS = 0.0f;
	float fNeighbUpdtTimeSum_MS = 0.0f;
	float fVarUpdtTimeSum_MS = 0.0f;
	float fInterKPsTimeSum_MS = 0.0f;
	float fIntraKPsTimeSum_MS = 0.0f;
	float fTotalKPsTime_MS = 0.0f;
	std::chrono::high_resolution_clock::time_point pre_all = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point post_lastKP = std::chrono::high_resolution_clock::now();
#endif //USE_INTERNAL_HRCS
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ldict = idx_uchar*m_nLocalWords;
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			size_t nMinColorDist = s_nColorMaxDataRange_1ch;
			size_t nMinDescDist = s_nDescMaxDataRange_1ch;
			size_t nMinSumDist = s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_burst = ((float*)(m_oMeanMinDistFrame_burst.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanLastDist_burst = ((float*)(m_oMeanLastDistFrame_burst.data+idx_flt32));
			float* pfCurrMeanRawSegmRes = ((float*)(m_oMeanRawSegmResFrame.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_burst = ((float*)(m_oMeanRawSegmResFrame_burst.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes = ((float*)(m_oMeanFinalSegmResFrame.data+idx_flt32));
			const float fBestLocalWordWeight = GetLocalWordWeight(m_aapLocalDicts[idx_ldict],m_nFrameIndex);
			const float fLocalWordsWeightSumThreshold = fBestLocalWordWeight/((*pfCurrDistThresholdFactor)*2);
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			const float fCurrGradBoostProp = (1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))/2;
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrColorDistThreshold = (size_t)((((*pfCurrDistThresholdFactor)*m_nColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*UNSTAB_COLOR_DIST_OFFSET))*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			const size_t nLastColorDist = absdiff_uchar(nLastColor,nCurrColor);
			const size_t nLastIntraDescDist = hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc);
			const float fNormalizedLastDist = ((float)nLastColorDist/s_nColorMaxDataRange_1ch+(float)nLastIntraDescDist/s_nDescMaxDataRange_1ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor) + fNormalizedLastDist*fRollAvgFactor;
			*pfCurrMeanLastDist_burst = (*pfCurrMeanLastDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedLastDist*fRollAvgFactor_burst;
			m_oGhostRegionMask.data[idx_uchar] = (	  (((*pfCurrMeanRawSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN1 || (*pfCurrMeanFinalSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN1) && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_DLST_MAX1 && (*pfCurrMeanLastDist_burst)<BGSCBLBSP_GHOST_DETECTION_DLST_MAX1)
												   || (*pfCurrMeanRawSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN2 || (*pfCurrMeanFinalSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN2
												 )?((m_oGhostRegionMask.data[idx_uchar]<UCHAR_MAX)?m_oGhostRegionMask.data[idx_uchar]+1:UCHAR_MAX):0;
			m_oHighVarRegionMask.data[idx_uchar] = (	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN1 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN1)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN2)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN3 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN3)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN4 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN4)
													||	((*pfCurrMeanRawSegmRes_burst)>BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN1 && (*pfCurrMeanLastDist_burst)>BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN1)
													||	((*pfCurrMeanRawSegmRes_burst)>BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN2 && (*pfCurrMeanLastDist_burst)>BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN2)
												   )?1:0;
			m_oUnstableRegionMask.data[idx_uchar] = (m_nFrameIndex<BOOTSTRAP_WIN_SIZE || (*pfCurrDistThresholdFactor)>BGSCBLBSP_INSTBLTY_DETECTION_MIN_R_VAL || (*pfCurrMeanRawSegmRes-*pfCurrMeanFinalSegmRes)>BGSCBLBSP_INSTBLTY_DETECTION_SEGM_DIFF || (!m_oFGMask_last.data[idx_uchar] && m_oFGMask_last2.data[idx_uchar]))?1:0;
			if(m_oIllumUpdtRegionMask.data[idx_uchar]) m_oIllumUpdtRegionMask.data[idx_uchar] = m_oIllumUpdtRegionMask.data[idx_uchar]-1;
			const size_t nCurrWordOccIncr = std::max((size_t)m_oGhostRegionMask.data[idx_uchar],nDefaultWordOccIncr);
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			float fLastLocalWordWeight = FLT_MAX;
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				const float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,pCurrLocalWord->nColor);
					size_t nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,pCurrLocalWord->nDesc);
					LBSP::computeGrayscaleDescriptor(oInputImg,pCurrLocalWord->nColor,x,y,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->nColor],nCurrInterDesc);
					size_t nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,pCurrLocalWord->nDesc);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					const size_t nSumDist = std::min((size_t)(fCurrGradBoostProp*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if( (*pfCurrDistThresholdFactor)<BGSCBLBSP_INSTBLTY_DETECTION_MIN_R_VAL
							&& nColorDist>=(3*nCurrColorDistThreshold)/4 && nIntraDescDist<=nCurrDescDistThreshold/4
							&& nColorDist<=(5*nCurrColorDistThreshold)/4
							&& (m_oIllumUpdtRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
						pCurrLocalWord->nColor = nCurrColor;
						m_oIllumUpdtRegionMask.data[idx_uchar] = ILLUMUPDT_REGION_DEFAULT_VAL;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
					if(nDescDist<=nCurrDescDistThreshold && nSumDist<=nCurrColorDistThreshold) {
						fPotentialLocalWordsWeightSum += fCurrLocalWordWeight;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						if((!m_oFGMask_last.data[idx_uchar] || m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar]) && fCurrLocalWordWeight<LWORD_MAX_WEIGHT)
							pCurrLocalWord->nOccurrences += nCurrWordOccIncr;
						if(nMinColorDist>nColorDist)
							nMinColorDist = nColorDist;
						if(nMinDescDist>nDescDist)
							nMinDescDist = nDescDist;
						if(nMinSumDist>nSumDist)
							nMinSumDist = nSumDist;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
				//failedcheck1ch:
				if(fCurrLocalWordWeight>fLastLocalWordWeight) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				else
					fLastLocalWordWeight = fCurrLocalWordWeight;
				++nLocalWordIdx;
			}
			while(nLocalWordIdx<m_nLocalWords) {
				const float fCurrLocalWordWeight = GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex);
				if(fCurrLocalWordWeight>fLastLocalWordWeight) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				else
					fLastLocalWordWeight = fCurrLocalWordWeight;
				++nLocalWordIdx;
			}
			if(fPotentialLocalWordsWeightSum>=fLocalWordsWeightSumThreshold) {
				// == background
				const float fNormalizedMinDist = (float)nMinSumDist/s_nColorMaxDataRange_1ch;
				*pfCurrMeanMinDist = (*pfCurrMeanMinDist)*(1.0f-fRollAvgFactor) + fNormalizedMinDist*fRollAvgFactor;
				*pfCurrMeanMinDist_burst = (*pfCurrMeanMinDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedMinDist*fRollAvgFactor_burst;
				*pfCurrMeanRawSegmRes = (*pfCurrMeanRawSegmRes)*(1.0f-fRollAvgFactor);
				*pfCurrMeanRawSegmRes_burst = (*pfCurrMeanRawSegmRes_burst)*(1.0f-fRollAvgFactor_burst);
				if(m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0) {
					const uchar nCurrIntraDescBITS = popcount_ushort_8bitsLUT(nCurrIntraDesc);
					GlobalWord_1ch* pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalWordLookupTable_BG[idx_uchar];
					if(!pLastMatchedGlobalWord
							|| absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold
							|| absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold/2) {
						size_t nGlobalWordIdx;
						for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold
									&& absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold/2)
								break;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							pLastMatchedGlobalWord->nColor = nCurrColor;
							pLastMatchedGlobalWord->nDescBITS = nCurrIntraDescBITS;
							pLastMatchedGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pLastMatchedGlobalWord->fLatestWeight = 0.0f;
						}
						m_apGlobalWordLookupTable_BG[idx_uchar] = pLastMatchedGlobalWord;
					}
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
			}
			else {
				// == foreground
				const float fNormalizedMinDist = std::min((float)nMinSumDist/s_nColorMaxDataRange_1ch+(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum)/fLocalWordsWeightSumThreshold,1.0f);
				*pfCurrMeanMinDist = (*pfCurrMeanMinDist)*(1.0f-fRollAvgFactor) + fNormalizedMinDist*fRollAvgFactor;
				*pfCurrMeanMinDist_burst = (*pfCurrMeanMinDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedMinDist*fRollAvgFactor_burst;
				*pfCurrMeanRawSegmRes = (*pfCurrMeanRawSegmRes)*(1.0f-fRollAvgFactor) + fRollAvgFactor;
				*pfCurrMeanRawSegmRes_burst = (*pfCurrMeanRawSegmRes_burst)*(1.0f-fRollAvgFactor_burst) + fRollAvgFactor_burst;
				const uchar nCurrIntraDescBITS = popcount_ushort_8bitsLUT(nCurrIntraDesc);
				GlobalWord_1ch* pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalWordLookupTable_FG[idx_uchar];
				if(!pLastMatchedGlobalWord
						|| absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold
						|| absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold/2) {
					size_t nGlobalWordIdx;
					for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
						pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
						if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold
								&& absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold/2)
							break;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable_FG[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || (*(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32))/2+fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold)
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
#if DISPLAY_CBLBSP_DEBUG_INFO
				else if(y==nDebugCoordY && x==nDebugCoordX) {
					bDBGMaskModifiedByGDict = true;
					pDBGGlobalWordModifier = pLastMatchedGlobalWord;
					fDBGGlobalWordModifierLocalWeight = *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
				}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
					const size_t nNewLocalWordIdx = m_nLocalWords-1;
					LocalWord_1ch* pNewLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
					pNewLocalWord->nColor = nCurrColor;
					pNewLocalWord->nDesc = nCurrIntraDesc;
					pNewLocalWord->nOccurrences = nCurrWordOccIncr;
					pNewLocalWord->nFirstOcc = m_nFrameIndex;
					pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
					vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				if(pLastMatchedGlobalWord && !m_oFGMask_last.data[idx_uchar] && (m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
			}
			// == neighb updt
			if((!oCurrFGMask.data[idx_uchar] || !m_oFGMask_last.data[idx_uchar]) && (m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
				int x_rand,y_rand;
				getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
				const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
				if(m_aapLocalDicts[idx_rand_ldict]) {
					size_t nRandLocalWordIdx = 0;
					float fPotentialRandLocalWordsWeightSum = 0.0f;
					while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
						LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
						const size_t nRandColorDist = absdiff_uchar(nCurrColor,pRandLocalWord->nColor);
						const size_t nRandIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc);
						if(nRandColorDist<=nCurrColorDistThreshold && nRandIntraDescDist<=nCurrDescDistThreshold) {
							const float fRandLocalWordWeight = GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
							fPotentialRandLocalWordsWeightSum += fRandLocalWordWeight;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
							if(fRandLocalWordWeight<LWORD_MAX_WEIGHT)
								pRandLocalWord->nOccurrences += nCurrWordOccIncr;
#if DISPLAY_CBLBSP_DEBUG_INFO
							vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
						}
						++nRandLocalWordIdx;
					}
					if(fPotentialRandLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						nRandLocalWordIdx = m_nLocalWords-1;
						LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
						pRandLocalWord->nColor = nCurrColor;
						pRandLocalWord->nDesc = nCurrIntraDesc;
						pRandLocalWord->nOccurrences = nCurrWordOccIncr;
						pRandLocalWord->nFirstOcc = m_nFrameIndex;
						pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
			}
			if(m_oFGMask_last.data[idx_uchar] && (*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
				*pfCurrLearningRate += BGSCBLBSP_T_INCR/std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst);
				if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
					*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
			}
			else if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
				*pfCurrLearningRate -= BGSCBLBSP_T_DECR/std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst);
				if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
					*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
			}
			if((std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]) || m_oHighVarRegionMask.data[idx_uchar]) {
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
				}
			}
			else if((*pfCurrDistThresholdVariationFactor)>BGSCBLBSP_R2_LOWER) {
				(*pfCurrDistThresholdVariationFactor) -= (m_oFGMask_last.data[idx_uchar]||m_oUnstableRegionMask.data[idx_uchar])?BGSCBLBSP_R2_DECR/8:BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) = BGSCBLBSP_R2_LOWER;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(BGSCBLBSP_R_LOWER+std::min(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst)*2,2)) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor-BGSCBLBSP_R2_LOWER);
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
#if DISPLAY_CBLBSP_DEBUG_INFO
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = nCurrColor;
					anDBGIntraDesc[c] = nCurrIntraDesc;
				}
				fDBGLocalWordsWeightSumThreshold = fLocalWordsWeightSumThreshold;
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
				nDBGWordOccIncr = std::max(nDBGWordOccIncr,nCurrWordOccIncr);
			}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		}
	}
	else { //m_nImgChannels==3
#if USE_INTERNAL_HRCS
		std::chrono::high_resolution_clock::time_point pre_loop = std::chrono::high_resolution_clock::now();
#endif //USE_INTERNAL_HRCS
		for(size_t k=0; k<nKeyPoints; ++k) {
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point pre_currKP = std::chrono::high_resolution_clock::now();
			fInterKPsTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(pre_currKP-post_lastKP).count())/1000000;
			std::chrono::high_resolution_clock::time_point pre_prep = std::chrono::high_resolution_clock::now();
#endif //USE_INTERNAL_HRCS
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ldict = idx_uchar*m_nLocalWords;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			size_t nMinTotColorDist = s_nColorMaxDataRange_3ch;
			size_t nMinTotDescDist = s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist = s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_burst = ((float*)(m_oMeanMinDistFrame_burst.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanLastDist_burst = ((float*)(m_oMeanLastDistFrame_burst.data+idx_flt32));
			float* pfCurrMeanRawSegmRes = ((float*)(m_oMeanRawSegmResFrame.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_burst = ((float*)(m_oMeanRawSegmResFrame_burst.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes = ((float*)(m_oMeanFinalSegmResFrame.data+idx_flt32));
			const float fBestLocalWordWeight = GetLocalWordWeight(m_aapLocalDicts[idx_ldict],m_nFrameIndex);
			const float fLocalWordsWeightSumThreshold = fBestLocalWordWeight/((*pfCurrDistThresholdFactor)*2);
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			const float fCurrGradBoostProp = (1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))/2;
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*UNSTAB_COLOR_DIST_OFFSET));
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = (3*nCurrTotColorDistThreshold)/4;
			const size_t nCurrSCDescDistThreshold = (3*nCurrTotDescDistThreshold)/4;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			const size_t nLastTotColorDist = L1dist_uchar(anLastColor,anCurrColor);
			const size_t nLastTotIntraDescDist = hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc);
			const float fNormalizedLastDist = ((float)nLastTotColorDist/s_nColorMaxDataRange_3ch+(float)nLastTotIntraDescDist/s_nDescMaxDataRange_3ch)/2;
			*pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor) + fNormalizedLastDist*fRollAvgFactor;
			*pfCurrMeanLastDist_burst = (*pfCurrMeanLastDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedLastDist*fRollAvgFactor_burst;
			m_oGhostRegionMask.data[idx_uchar] = (	  (((*pfCurrMeanRawSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN1 || (*pfCurrMeanFinalSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN1) && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_DLST_MAX1 && (*pfCurrMeanLastDist_burst)<BGSCBLBSP_GHOST_DETECTION_DLST_MAX1)
												   || (*pfCurrMeanRawSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN2 || (*pfCurrMeanFinalSegmRes)>BGSCBLBSP_GHOST_DETECTION_SAVG_MIN2
												 )?((m_oGhostRegionMask.data[idx_uchar]<UCHAR_MAX)?m_oGhostRegionMask.data[idx_uchar]+1:UCHAR_MAX):0;
			m_oHighVarRegionMask.data[idx_uchar] = (	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN1 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN1)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN2)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN3 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN3)
													||	((*pfCurrMeanRawSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN4 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN4)
													||	((*pfCurrMeanRawSegmRes_burst)>BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN1 && (*pfCurrMeanLastDist_burst)>BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN1)
													||	((*pfCurrMeanRawSegmRes_burst)>BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN2 && (*pfCurrMeanLastDist_burst)>BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN2)
												   )?1:0;
			m_oUnstableRegionMask.data[idx_uchar] = (m_nFrameIndex<BOOTSTRAP_WIN_SIZE || (*pfCurrDistThresholdFactor)>BGSCBLBSP_INSTBLTY_DETECTION_MIN_R_VAL || (*pfCurrMeanRawSegmRes-*pfCurrMeanFinalSegmRes)>BGSCBLBSP_INSTBLTY_DETECTION_SEGM_DIFF || (!m_oFGMask_last.data[idx_uchar] && m_oFGMask_last2.data[idx_uchar]))?1:0;
			if(m_oIllumUpdtRegionMask.data[idx_uchar]) m_oIllumUpdtRegionMask.data[idx_uchar] = m_oIllumUpdtRegionMask.data[idx_uchar]-1;
			const size_t nCurrWordOccIncr = std::max((size_t)m_oGhostRegionMask.data[idx_uchar],nDefaultWordOccIncr);
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			float fLastLocalWordWeight = FLT_MAX;
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point post_prep = std::chrono::high_resolution_clock::now();
			fPrepTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_prep-pre_prep).count())/1000000;
#endif //USE_INTERNAL_HRCS
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				const float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
				{
#if USE_SC_THRS_VALIDATION
					size_t nTotColorDist = 0;
					size_t nTotIntraDescDist = 0;
					size_t nTotDescDist = 0;
					size_t nTotSumDist = 0;
					for(size_t c=0;c<3; ++c) {
						const size_t nColorDist = absdiff_uchar(anCurrColor[c],pCurrLocalWord->anColor[c]);
						if(nColorDist>nCurrSCColorDistThreshold)
							goto failedcheck3ch;
						size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],pCurrLocalWord->anDesc[c]);
#if USE_HARD_SC_DESC_DIST_CHECKS
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
#endif //USE_HARD_SC_DESC_DIST_CHECKS
						LBSP::computeSingleRGBDescriptor(oInputImg,pCurrLocalWord->anColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[c]],anCurrInterDesc[c]);
						size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],pCurrLocalWord->anDesc[c]);
#if USE_HARD_SC_DESC_DIST_CHECKS
						if(nInterDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
#endif //USE_HARD_DESC_DIST_CHECKS
						const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
#if !USE_HARD_SC_DESC_DIST_CHECKS
						if(nDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
#endif //!USE_HARD_SC_DESC_DIST_CHECKS
						const size_t nSumDist = std::min((size_t)(fCurrGradBoostProp*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
						if(nSumDist>nCurrSCColorDistThreshold)
							goto failedcheck3ch;
						nTotColorDist += nColorDist;
						nTotIntraDescDist += nIntraDescDist;
						nTotDescDist += nDescDist;
						nTotSumDist += nSumDist;
					}
#endif //USE_SC_THRS_VALIDATION
					if( (*pfCurrDistThresholdFactor)<BGSCBLBSP_INSTBLTY_DETECTION_MIN_R_VAL
							&& nTotColorDist>=(3*nCurrTotColorDistThreshold)/4
							&& nTotIntraDescDist<=nCurrTotDescDistThreshold/4
							&& (nTotColorDist/2+cdist_uchar(anCurrColor,pCurrLocalWord->anColor)*6)<=nCurrTotColorDistThreshold
							&& (m_oIllumUpdtRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
						for(size_t c=0; c<3; ++c)
							pCurrLocalWord->anColor[c] = anCurrColor[c];
						m_oIllumUpdtRegionMask.data[idx_uchar] = ILLUMUPDT_REGION_DEFAULT_VAL;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
#if !USE_SC_THRS_VALIDATION
					const size_t nTotColorDist = L1dist_uchar(anCurrColor,pCurrLocalWord->anColor);
					if(nTotColorDist>nCurrTotColorDistThreshold)
						goto failedcheck3ch;
					const size_t nTotIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc,pCurrLocalWord->anDesc);
					const size_t anCurrInterLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[0]],m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[1]],m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[2]]};
					LBSP::computeRGBDescriptor(oInputImg,pCurrLocalWord->anColor,x,y,anCurrInterLBSPThresholds,anCurrInterDesc);
					const size_t nTotInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc,pCurrLocalWord->anDesc);
					const size_t nTotDescDist = (nTotIntraDescDist+nTotInterDescDist)/2;
					if(nTotDescDist>nCurrTotDescDistThreshold)
						goto failedcheck3ch;
					const size_t nTotSumDist = std::min((size_t)(fCurrGradBoostProp*nTotDescDist)*(s_nColorMaxDataRange_3ch/s_nDescMaxDataRange_3ch)+nTotColorDist,s_nColorMaxDataRange_3ch);
#endif //!USE_SC_THRS_VALIDATION
					if(nTotDescDist<=nCurrTotDescDistThreshold && nTotSumDist<=nCurrTotColorDistThreshold) {
						fPotentialLocalWordsWeightSum += fCurrLocalWordWeight;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
						if((!m_oFGMask_last.data[idx_uchar] || m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar]) && fCurrLocalWordWeight<LWORD_MAX_WEIGHT)
							pCurrLocalWord->nOccurrences += nCurrWordOccIncr;
						if(nMinTotColorDist>nTotColorDist)
							nMinTotColorDist = nTotColorDist;
						if(nMinTotDescDist>nTotDescDist)
							nMinTotDescDist = nTotDescDist;
						if(nMinTotSumDist>nTotSumDist)
							nMinTotSumDist = nTotSumDist;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
				failedcheck3ch:
				if(fCurrLocalWordWeight>fLastLocalWordWeight) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				else
					fLastLocalWordWeight = fCurrLocalWordWeight;
				++nLocalWordIdx;
			}
			while(nLocalWordIdx<m_nLocalWords) {
				const float fCurrLocalWordWeight = GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex);
				if(fCurrLocalWordWeight>fLastLocalWordWeight) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				else
					fLastLocalWordWeight = fCurrLocalWordWeight;
				++nLocalWordIdx;
			}
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point post_ldictscan = std::chrono::high_resolution_clock::now();
			fLDictScanTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_ldictscan-post_prep).count())/1000000;
#endif //USE_INTERNAL_HRCS
			if(fPotentialLocalWordsWeightSum>=fLocalWordsWeightSumThreshold) {
				// == background
				const float fNormalizedMinDist = (float)nMinTotSumDist/s_nColorMaxDataRange_3ch;
				*pfCurrMeanMinDist = (*pfCurrMeanMinDist)*(1.0f-fRollAvgFactor) + fNormalizedMinDist*fRollAvgFactor;
				*pfCurrMeanMinDist_burst = (*pfCurrMeanMinDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedMinDist*fRollAvgFactor_burst;
				*pfCurrMeanRawSegmRes = (*pfCurrMeanRawSegmRes)*(1.0f-fRollAvgFactor);
				*pfCurrMeanRawSegmRes_burst = (*pfCurrMeanRawSegmRes_burst)*(1.0f-fRollAvgFactor_burst);
				if(m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0) {
					const uchar nCurrIntraDescBITS = popcount_ushort_8bitsLUT(anCurrIntraDesc);
					GlobalWord_3ch* pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalWordLookupTable_BG[idx_uchar];
					if(!pLastMatchedGlobalWord
							|| absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold/2
							|| (L1dist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)/2+cdist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)*6)>nCurrTotColorDistThreshold) {
						size_t nGlobalWordIdx;
						for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
							pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							if(absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold/2
									&& (L1dist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)/2+cdist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)*6)<=nCurrTotColorDistThreshold)
								break;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							for(size_t c=0; c<3; ++c)
								pLastMatchedGlobalWord->anColor[c] = anCurrColor[c];
							pLastMatchedGlobalWord->nDescBITS = nCurrIntraDescBITS;
							pLastMatchedGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pLastMatchedGlobalWord->fLatestWeight = 0.0f;
						}
						m_apGlobalWordLookupTable_BG[idx_uchar] = pLastMatchedGlobalWord;
					}
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
			}
			else {
				// == foreground
				const float fNormalizedMinDist = std::min((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum)/fLocalWordsWeightSumThreshold,1.0f);
				*pfCurrMeanMinDist = (*pfCurrMeanMinDist)*(1.0f-fRollAvgFactor) + fNormalizedMinDist*fRollAvgFactor;
				*pfCurrMeanMinDist_burst = (*pfCurrMeanMinDist_burst)*(1.0f-fRollAvgFactor_burst) + fNormalizedMinDist*fRollAvgFactor_burst;
				*pfCurrMeanRawSegmRes = (*pfCurrMeanRawSegmRes)*(1.0f-fRollAvgFactor) + fRollAvgFactor;
				*pfCurrMeanRawSegmRes_burst = (*pfCurrMeanRawSegmRes_burst)*(1.0f-fRollAvgFactor_burst) + fRollAvgFactor_burst;
				const uchar nCurrIntraDescBITS = popcount_ushort_8bitsLUT(anCurrIntraDesc);
				GlobalWord_3ch* pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalWordLookupTable_FG[idx_uchar];
				if(!pLastMatchedGlobalWord
						|| absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold/2
						|| (L1dist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)/2+cdist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)*6)>nCurrTotColorDistThreshold) {
					size_t nGlobalWordIdx;
					for(nGlobalWordIdx=0;nGlobalWordIdx<m_nGlobalWords;++nGlobalWordIdx) {
						pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
						if(absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold/2
								&& (L1dist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)/2+cdist_uchar(anCurrColor,pLastMatchedGlobalWord->anColor)*6)<=nCurrTotColorDistThreshold)
							break;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable_FG[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || (*(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32))/2+fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold)
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
#if DISPLAY_CBLBSP_DEBUG_INFO
				else if(y==nDebugCoordY && x==nDebugCoordX) {
					bDBGMaskModifiedByGDict = true;
					pDBGGlobalWordModifier = pLastMatchedGlobalWord;
					fDBGGlobalWordModifierLocalWeight = *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
				}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
					const size_t nNewLocalWordIdx = m_nLocalWords-1;
					LocalWord_3ch* pNewLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
					for(size_t c=0; c<3; ++c) {
						pNewLocalWord->anColor[c] = anCurrColor[c];
						pNewLocalWord->anDesc[c] = anCurrIntraDesc[c];
					}
					pNewLocalWord->nOccurrences = nCurrWordOccIncr;
					pNewLocalWord->nFirstOcc = m_nFrameIndex;
					pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
					vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				if(pLastMatchedGlobalWord && !m_oFGMask_last.data[idx_uchar] && (m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
			}
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point post_rawdecision = std::chrono::high_resolution_clock::now();
			if(oCurrFGMask.data[idx_uchar])
				fFGRawTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_rawdecision-post_ldictscan).count())/1000000;
			else
				fBGRawTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_rawdecision-post_ldictscan).count())/1000000;
#endif //USE_INTERNAL_HRCS
			// == neighb updt
			if((!oCurrFGMask.data[idx_uchar] || !m_oFGMask_last.data[idx_uchar]) && (m_oGhostRegionMask.data[idx_uchar] || m_oHighVarRegionMask.data[idx_uchar] || (rand()%nCurrLocalWordUpdateRate)==0)) {
				int x_rand,y_rand;
				getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
				const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
				if(m_aapLocalDicts[idx_rand_ldict]) {
					size_t nRandLocalWordIdx = 0;
					float fPotentialRandLocalWordsWeightSum = 0.0f;
					while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
						LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
						const size_t nRandTotColorDist = L1dist_uchar(anCurrColor,pRandLocalWord->anColor);
						const size_t nRandTotIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc);
						if(nRandTotColorDist<=nCurrTotColorDistThreshold && nRandTotIntraDescDist<=nCurrTotDescDistThreshold) {
							const float fRandLocalWordWeight = GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
							fPotentialRandLocalWordsWeightSum += fRandLocalWordWeight;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
							if(fRandLocalWordWeight<LWORD_MAX_WEIGHT)
								pRandLocalWord->nOccurrences += nCurrWordOccIncr;
#if DISPLAY_CBLBSP_DEBUG_INFO
							vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
						}
						++nRandLocalWordIdx;
					}
					if(fPotentialRandLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						nRandLocalWordIdx = m_nLocalWords-1;
						LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
						for(size_t c=0; c<3; ++c) {
							pRandLocalWord->anColor[c] = anCurrColor[c];
							pRandLocalWord->anDesc[c] = anCurrIntraDesc[c];
						}
						pRandLocalWord->nOccurrences = nCurrWordOccIncr;
						pRandLocalWord->nFirstOcc = m_nFrameIndex;
						pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
			}
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point post_neighbupdt = std::chrono::high_resolution_clock::now();
			fNeighbUpdtTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_neighbupdt-post_rawdecision).count())/1000000;
#endif //USE_INTERNAL_HRCS
			if(m_oFGMask_last.data[idx_uchar] && (*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
				*pfCurrLearningRate += BGSCBLBSP_T_INCR/std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst);
				if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
					*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
			}
			else if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
				*pfCurrLearningRate -= BGSCBLBSP_T_DECR/std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst);
				if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
					*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
			}
			if((std::max(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]) || m_oHighVarRegionMask.data[idx_uchar]) {
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
				}
			}
			else if((*pfCurrDistThresholdVariationFactor)>BGSCBLBSP_R2_LOWER) {
				(*pfCurrDistThresholdVariationFactor) -= (m_oFGMask_last.data[idx_uchar]||m_oUnstableRegionMask.data[idx_uchar])?BGSCBLBSP_R2_DECR/8:BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) = BGSCBLBSP_R2_LOWER;
			}
			if((*pfCurrDistThresholdFactor)<std::pow(BGSCBLBSP_R_LOWER+std::min(*pfCurrMeanMinDist,*pfCurrMeanMinDist_burst)*2,2)) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor-BGSCBLBSP_R2_LOWER);
					if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_LOWER;
			}
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
#if USE_INTERNAL_HRCS
			std::chrono::high_resolution_clock::time_point post_varupdt = std::chrono::high_resolution_clock::now();
			fVarUpdtTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_varupdt-post_neighbupdt).count())/1000000;
			post_lastKP = std::chrono::high_resolution_clock::now();
			fIntraKPsTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_lastKP-pre_currKP).count())/1000000;
#endif //USE_INTERNAL_HRCS
#if DISPLAY_CBLBSP_DEBUG_INFO
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = anCurrColor[c];
					anDBGIntraDesc[c] = anCurrIntraDesc[c];
				}
				fDBGLocalWordsWeightSumThreshold = fLocalWordsWeightSumThreshold;
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
				nDBGWordOccIncr = std::max(nDBGWordOccIncr,nCurrWordOccIncr);
			}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		}
#if USE_INTERNAL_HRCS
		std::chrono::high_resolution_clock::time_point post_loop = std::chrono::high_resolution_clock::now();
		fInterKPsTimeSum_MS += (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(post_loop-post_lastKP).count())/1000000;
		fTotalKPsTime_MS = (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_loop-pre_loop).count())/1000;
#endif //USE_INTERNAL_HRCS
	}
#if USE_INTERNAL_HRCS
	std::chrono::high_resolution_clock::time_point pre_gword_calcs = std::chrono::high_resolution_clock::now();
#endif //USE_INTERNAL_HRCS
	const bool bRecalcGlobalWords = !(m_nFrameIndex%(nCurrGlobalWordUpdateRate<<5));
	const bool bUpdateGlobalWords = !(m_nFrameIndex%(nCurrGlobalWordUpdateRate));
	for(size_t nGlobalWordIdx=0; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx) {
		if(bRecalcGlobalWords && m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>0.0f) {
			m_apGlobalDict[nGlobalWordIdx]->fLatestWeight = GetGlobalWordWeight(m_apGlobalDict[nGlobalWordIdx]);
			if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight<1.0f) {
				m_apGlobalDict[nGlobalWordIdx]->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap = cv::Scalar(0.0f);
			}
		}
		if(bUpdateGlobalWords && m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>0.0f) {
			cv::accumulateProduct(m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_oTempGlobalWordWeightDiffFactor,m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_oFGMask_last_dilated_inverted);
			m_apGlobalDict[nGlobalWordIdx]->fLatestWeight *= (1.0f-GWORD_WEIGHT_DECIMATION_FACTOR);
			cv::blur(m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,cv::Size(7,7),cv::Point(-1,-1),cv::BORDER_REPLICATE);
		}
		if(nGlobalWordIdx>0 && m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)
			std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
	}
#if USE_INTERNAL_HRCS
	std::chrono::high_resolution_clock::time_point post_gword_calcs = std::chrono::high_resolution_clock::now();
	std::cout << "t=" << m_nFrameIndex << " : ";
	std::cout << "*prep=" << std::fixed << std::setprecision(1) << fPrepTimeSum_MS << ", ";
	std::cout << "*ldict=" << std::fixed << std::setprecision(1) << fLDictScanTimeSum_MS << ", ";
	std::cout << "*decision=" << std::fixed << std::setprecision(1) << (fBGRawTimeSum_MS+fFGRawTimeSum_MS) << "(" << (int)((fBGRawTimeSum_MS/(fBGRawTimeSum_MS+fFGRawTimeSum_MS))*99.9f) << "%bg), ";
	std::cout << "*nghbupdt=" << std::fixed << std::setprecision(1) << fNeighbUpdtTimeSum_MS << ", ";
	std::cout << "*varupdt=" << std::fixed << std::setprecision(1) << fVarUpdtTimeSum_MS << ", ";
	std::cout << "kptsinter=" << std::fixed << std::setprecision(1) << fInterKPsTimeSum_MS << ", ";
	std::cout << "kptsintra=" << std::fixed << std::setprecision(1) << fIntraKPsTimeSum_MS << ", ";
	std::cout << "kptsloop=" << std::fixed << std::setprecision(1) << fTotalKPsTime_MS << ", ";
	std::cout << "gwordupdt=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_gword_calcs-pre_gword_calcs).count())/1000 << ", ";
#endif //USE_INTERNAL_HRCS
#if DISPLAY_CBLBSP_DEBUG_INFO
	if(idx_dbg_ldict!=UINT_MAX) {
		std::cout << std::endl;
		cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
		cv::Mat oGlobalWordsCoverageMap(m_oImgSize,CV_32FC1,cv::Scalar(0.0f));
		for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords; ++nDBGWordIdx)
			cv::max(oGlobalWordsCoverageMap,m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap,oGlobalWordsCoverageMap);
		cv::resize(oGlobalWordsCoverageMap,oGlobalWordsCoverageMap,DEBUG_WINDOW_SIZE);
		cv::imshow("oGlobalWordsCoverageMap",oGlobalWordsCoverageMap);
		/*std::string asDBGStrings[5] = {"gword[0]","gword[1]","gword[2]","gword[3]","gword[4]"};
		for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords && nDBGWordIdx<5; ++nDBGWordIdx)
			cv::imshow(asDBGStrings[nDBGWordIdx],m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap);
		double minVal,maxVal;
		cv::minMaxIdx(gwords_coverage,&minVal,&maxVal);
		std::cout << " " << m_nFrameIndex << " : gwords_coverage min=" << minVal << ", max=" << maxVal << std::endl;*/
		if(true) {
			printf("\nDBG[%2d,%2d] : \n",nDebugCoordX,nDebugCoordY);
			printf("\t Color=[%03d,%03d,%03d]\n",(int)anDBGColor[0],(int)anDBGColor[1],(int)anDBGColor[2]);
			printf("\t IntraDesc=[%05d,%05d,%05d], IntraDescBITS=[%02lu,%02lu,%02lu]\n",anDBGIntraDesc[0],anDBGIntraDesc[1],anDBGIntraDesc[2],(size_t)popcount_ushort_8bitsLUT(anDBGIntraDesc[0]),(size_t)popcount_ushort_8bitsLUT(anDBGIntraDesc[1]),(size_t)popcount_ushort_8bitsLUT(anDBGIntraDesc[2]));
			char gword_dbg_str[1024] = "\0";
			if(bDBGMaskModifiedByGDict) {
				if(m_nImgChannels==1) {
					GlobalWord_1ch* pDBGGlobalWordModifier_1ch = (GlobalWord_1ch*)pDBGGlobalWordModifier;
					sprintf(gword_dbg_str,"* aided by gword weight=[%02.03f], nColor=[%03d], nDescBITS=[%02lu]",fDBGGlobalWordModifierLocalWeight,(int)pDBGGlobalWordModifier_1ch->nColor,(size_t)pDBGGlobalWordModifier_1ch->nDescBITS);
				}
				else { //m_nImgChannels==3
					GlobalWord_3ch* pDBGGlobalWordModifier_3ch = (GlobalWord_3ch*)pDBGGlobalWordModifier;
					sprintf(gword_dbg_str,"* aided by gword weight=[%02.03f], anColor=[%03d,%03d,%03d], nDescBITS=[%02lu]",fDBGGlobalWordModifierLocalWeight,(int)pDBGGlobalWordModifier_3ch->anColor[0],(int)pDBGGlobalWordModifier_3ch->anColor[1],(int)pDBGGlobalWordModifier_3ch->anColor[2],(size_t)pDBGGlobalWordModifier_3ch->nDescBITS);
				}
			}
			printf("\t FG_Mask=[%s] %s\n",(bDBGMaskResult?"TRUE":"FALSE"),gword_dbg_str);
			printf("----\n");
			printf("DBG_LDICT : (%lu occincr per match)\n",nDBGWordOccIncr);
			for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nLocalWords; ++nDBGWordIdx) {
				if(m_nImgChannels==1) {
					LocalWord_1ch* pDBGLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_dbg_ldict+nDBGWordIdx];
					printf("\t [%02lu] : weight=[%02.03f], nColor=[%03d], nDescBITS=[%02lu]  %s\n",nDBGWordIdx,GetLocalWordWeight(pDBGLocalWord,m_nFrameIndex),(int)pDBGLocalWord->nColor,(size_t)popcount_ushort_8bitsLUT(pDBGLocalWord->nDesc),vsWordModList[idx_dbg_ldict+nDBGWordIdx].c_str());
				}
				else { //m_nImgChannels==3
					LocalWord_3ch* pDBGLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_dbg_ldict+nDBGWordIdx];
					printf("\t [%02lu] : weight=[%02.03f], anColor=[%03d,%03d,%03d], anDescBITS=[%02lu,%02lu,%02lu]  %s\n",nDBGWordIdx,GetLocalWordWeight(pDBGLocalWord,m_nFrameIndex),(int)pDBGLocalWord->anColor[0],(int)pDBGLocalWord->anColor[1],(int)pDBGLocalWord->anColor[2],(size_t)popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[0]),(size_t)popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[1]),(size_t)popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[2]),vsWordModList[idx_dbg_ldict+nDBGWordIdx].c_str());
				}
			}
		}
		std::cout << std::fixed << std::setprecision(5) << " w_thrs(" << dbgpt << ") = " << fDBGLocalWordsWeightSumThreshold << std::endl;
		cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
		cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanMinDistFrameNormalized_burst; m_oMeanMinDistFrame_burst.copyTo(oMeanMinDistFrameNormalized_burst);
		cv::circle(oMeanMinDistFrameNormalized_burst,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanMinDistFrameNormalized_burst,oMeanMinDistFrameNormalized_burst,DEBUG_WINDOW_SIZE);
		cv::imshow("d_min_burst(x)",oMeanMinDistFrameNormalized_burst);
		std::cout << std::fixed << std::setprecision(5) << " d_min2(" << dbgpt << ") = " << m_oMeanMinDistFrame_burst.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
		cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanLastDistFrameNormalized_burst; m_oMeanLastDistFrame_burst.copyTo(oMeanLastDistFrameNormalized_burst);
		cv::circle(oMeanLastDistFrameNormalized_burst,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanLastDistFrameNormalized_burst,oMeanLastDistFrameNormalized_burst,DEBUG_WINDOW_SIZE);
		cv::imshow("d_last_burst(x)",oMeanLastDistFrameNormalized_burst);
		std::cout << std::fixed << std::setprecision(5) << " d_lst2(" << dbgpt << ") = " << m_oMeanLastDistFrame_burst.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame.copyTo(oMeanRawSegmResFrameNormalized);
		cv::circle(oMeanRawSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanRawSegmResFrameNormalized_burst; m_oMeanRawSegmResFrame_burst.copyTo(oMeanRawSegmResFrameNormalized_burst);
		cv::circle(oMeanRawSegmResFrameNormalized_burst,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanRawSegmResFrameNormalized_burst,oMeanRawSegmResFrameNormalized_burst,DEBUG_WINDOW_SIZE);
		cv::imshow("s_avg_burst(x)",oMeanRawSegmResFrameNormalized_burst);
		std::cout << std::fixed << std::setprecision(5) << " s_avg2(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_burst.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame.copyTo(oMeanFinalSegmResFrameNormalized);
		cv::circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,1.0f/BGSCBLBSP_R_UPPER,-BGSCBLBSP_R_LOWER/BGSCBLBSP_R_UPPER);
		cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("r(x)",oDistThresholdFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oDistThresholdVariationFrameNormalized; cv::normalize(m_oDistThresholdVariationFrame,oDistThresholdVariationFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
		cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oDistThresholdVariationFrameNormalized,oDistThresholdVariationFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "     r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSCBLBSP_T_UPPER,-BGSCBLBSP_T_LOWER/BGSCBLBSP_T_UPPER);
		cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("t(x)",oUpdateRateFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
		cv::imshow("s(x)",oCurrFGMask);
		cv::Mat oHighVarRegionMaskNormalized; m_oHighVarRegionMask.copyTo(oHighVarRegionMaskNormalized); oHighVarRegionMaskNormalized*=UCHAR_MAX;
		cv::circle(oHighVarRegionMaskNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oHighVarRegionMaskNormalized,oHighVarRegionMaskNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("m_oHighVarRegionMask",oHighVarRegionMaskNormalized);
		cv::Mat oGhostRegionMaskNormalized; m_oGhostRegionMask.copyTo(oGhostRegionMaskNormalized);
		cv::circle(oGhostRegionMaskNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oGhostRegionMaskNormalized,oGhostRegionMaskNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("m_oGhostRegionMask",oGhostRegionMaskNormalized);
		cv::Mat oUnstableRegionMaskNormalized; m_oUnstableRegionMask.copyTo(oUnstableRegionMaskNormalized); oUnstableRegionMaskNormalized*=UCHAR_MAX;
		cv::circle(oUnstableRegionMaskNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oUnstableRegionMaskNormalized,oUnstableRegionMaskNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("m_oUnstableRegionMask",oUnstableRegionMaskNormalized);
		cv::Mat oIllumUpdtRegionMaskNormalized; m_oIllumUpdtRegionMask.copyTo(oIllumUpdtRegionMaskNormalized); oIllumUpdtRegionMaskNormalized*=(UCHAR_MAX/ILLUMUPDT_REGION_DEFAULT_VAL);
		cv::circle(oIllumUpdtRegionMaskNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oIllumUpdtRegionMaskNormalized,oIllumUpdtRegionMaskNormalized,DEBUG_WINDOW_SIZE);
		cv::imshow("m_oIllumUpdtRegionMask",oIllumUpdtRegionMaskNormalized);
	}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
	cv::bitwise_xor(oCurrFGMask,m_oPureFGMask_last,m_oPureFGBlinkMask_curr);
	cv::bitwise_or(m_oPureFGBlinkMask_curr,m_oPureFGBlinkMask_last,m_oBlinksFrame);
	m_oPureFGBlinkMask_curr.copyTo(m_oPureFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oPureFGMask_last);
	//cv::imshow("orig raw",oCurrFGMask);
	cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	//cv::imshow("post-1close",m_oFGMask_PreFlood);
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	//cv::imshow("post-1close, flooded+inverted",m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),2);
	//cv::imshow("post-1close, 2eroded",m_oFGMask_PreFlood);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
	m_oFGMask_last.copyTo(m_oFGMask_last2);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
	//cv::imshow("result",m_oFGMask_last);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
	cv::addWeighted(m_oMeanFinalSegmResFrame,(1.0f-fRollAvgFactor),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor,0,m_oMeanFinalSegmResFrame,CV_32F);
	const float fFinalFGRatio = (float)cv::sum(m_oFGMask_last).val[0]/(nKeyPoints*256);
	if(fFinalFGRatio>MODEL_RESET_MIN_FINAL_FG_RATIO) {
		++m_nModelResetFrameCount;
		if(m_nModelResetFrameCount>=MODEL_RESET_MIN_FRAME_COUNT)
			refreshModel(1,LWORD_WEIGHT_OFFSET/2,0.5f,true);
	}
	else if(m_nModelResetFrameCount)
		m_nModelResetFrameCount = 0;
#if USE_INTERNAL_HRCS
	std::chrono::high_resolution_clock::time_point post_morphops = std::chrono::high_resolution_clock::now();
	std::cout << "morphops=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_morphops-post_gword_calcs).count())/1000 << ", ";
	std::chrono::high_resolution_clock::time_point post_all = std::chrono::high_resolution_clock::now();
	std::cout << "all=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_all-pre_all).count())/1000 << ". " << std::endl;
#endif //USE_INTERNAL_HRCS
#if USE_INTERNAL_RCHECKS
	size_t end_rss = getCurrentRSS();
	std::cout << "t=" << m_nFrameIndex << " : memdiff start/end=" << abs((int)end_rss-(int)start_rss) << std::endl;
#endif //USE_INTERNAL_RCHECKS
}

void BackgroundSubtractorCBLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	const size_t nKeyPoints = m_voKeyPoints.size();
	for(size_t k=0; k<nKeyPoints; ++k) {
		const int x = (int)m_voKeyPoints[k].pt.x;
		const int y = (int)m_voKeyPoints[k].pt.y;
		const size_t idx_uchar = m_oImgSize.width*y + x;
		const size_t idx_ldict = idx_uchar*m_nLocalWords;
		if(m_nImgChannels==1) {
			float fTotWeight = 0.0f;
			float fTotColor = 0.0f;
			for(size_t n=0; n<m_nLocalWords; ++n) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict];
				float fCurrWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
				fTotColor += (float)pCurrLocalWord->nColor*fCurrWeight;
				fTotWeight += fCurrWeight;
			}
			oAvgBGImg.at<float>(y,x) = fTotColor/fTotWeight;
		}
		else { //m_nImgChannels==3
			float fTotWeight = 0.0f;
			float fTotColor[3] = {0.0f,0.0f,0.0f};
			for(size_t n=0; n<m_nLocalWords; ++n) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict];
				float fCurrWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
				for(size_t c=0; c<3; ++c)
					fTotColor[c] += (float)pCurrLocalWord->anColor[c]*fCurrWeight;
				fTotWeight += fCurrWeight;
			}
			oAvgBGImg.at<cv::Vec3f>(y,x) = cv::Vec3f(fTotColor[0]/fTotWeight,fTotColor[1]/fTotWeight,fTotColor[2]/fTotWeight);
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

void BackgroundSubtractorCBLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_Assert(LBSP::DESC_SIZE==2);
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	// @@@@@@ TO BE REWRITTEN FOR WORD-BASED RECONSTRUCTION
	/*for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
				const size_t idx_flt32 = idx_ndesc*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+idx_flt32);
				const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
			}
		}
	}*/
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

void BackgroundSubtractorCBLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	CV_Assert(!m_bInitializedInternalStructs);
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
}

void BackgroundSubtractorCBLBSP::CleanupDictionaries() {
	if(m_apLocalWordList_1ch) {
		delete[] m_apLocalWordList_1ch;
		m_apLocalWordList_1ch = nullptr;
	}
	if(m_apLocalWordList_3ch) {
		delete[] m_apLocalWordList_3ch;
		m_apLocalWordList_3ch = nullptr;
	}
	if(m_aapLocalDicts) {
		delete[] m_aapLocalDicts;
		m_aapLocalDicts = nullptr;
	}
	if(m_apGlobalWordList_1ch) {
		delete[] m_apGlobalWordList_1ch;
		m_apGlobalWordList_1ch = nullptr;
	}
	if(m_apGlobalWordList_3ch) {
		delete[] m_apGlobalWordList_3ch;
		m_apGlobalWordList_3ch = nullptr;
	}
	if(m_apGlobalDict) {
		delete[] m_apGlobalDict;
		m_apGlobalDict = nullptr;
	}
	if(m_apGlobalWordLookupTable_BG) {
		delete[] m_apGlobalWordLookupTable_BG;
		m_apGlobalWordLookupTable_BG = nullptr;
	}
	if(m_apGlobalWordLookupTable_FG) {
		delete[] m_apGlobalWordLookupTable_FG;
		m_apGlobalWordLookupTable_FG = nullptr;
	}
}

float BackgroundSubtractorCBLBSP::GetLocalWordWeight(const LocalWord* w, size_t nCurrFrame) {
	return (float)(w->nOccurrences)/((w->nLastOcc-w->nFirstOcc)/2+(nCurrFrame-w->nLastOcc)+LWORD_WEIGHT_OFFSET);
}

float BackgroundSubtractorCBLBSP::GetGlobalWordWeight(const GlobalWord* w) {
	return (float)cv::sum(w->oSpatioOccMap).val[0];
}

BackgroundSubtractorCBLBSP::LocalWord::~LocalWord() {}

BackgroundSubtractorCBLBSP::GlobalWord::~GlobalWord() {}
