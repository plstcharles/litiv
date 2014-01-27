#include "BackgroundSubtractorCBLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// @@@@@@ FOR FEEDBACK LOOPS, VARYING WEIGHT THRESHOLD > VARYING DIST THRESHOLD

// local define used for debug purposes only
#define DISPLAY_CBLBSP_DEBUG_INFO 1
// local define used to specify whether to use global words or not
#define USE_GLOBAL_WORDS 1
// local define for the gradient proportion value used in color+grad distance calculations
#define OVERLOAD_GRAD_PROP ((1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))*0.5f)
// local define for the lword representation update rate
#define LWORD_REPRESENTATION_UPDATE_RATE 16
// local define for the gword representation update rate
#define GWORD_REPRESENTATION_UPDATE_RATE 4
// local define for potential local word weight sum threshold
//#define LWORD_WEIGHT_SUM_THRESHOLD 1.0f
// local define for potential global word weight sum threshold
#define GWORD_WEIGHT_SUM_THRESHOLD 1.0f
// local define for the gword decimation factor
#define GWORD_WEIGHT_DECIMATION_FACTOR 0.9f
// local define for the amount of weight offset to apply to words, making sure new words aren't always better than old ones
#define LWORD_WEIGHT_OFFSET 750
// local define for the initial weight of a new word (used to make sure old words aren't worse off than new seeds)
#define LWORD_INIT_WEIGHT (1.0f/LWORD_WEIGHT_OFFSET)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nDescDistTypeCutoff_1ch = s_nDescMaxDataRange_1ch/2;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;
static const size_t s_nDescDistTypeCutoff_3ch = s_nDescDistTypeCutoff_1ch*3;

BackgroundSubtractorCBLBSP::BackgroundSubtractorCBLBSP(	 float fLBSPThreshold
														,size_t nLBSPThresholdOffset
														,size_t nInitDescDistThreshold
														,size_t nInitColorDistThreshold
														,float fLocalWordsPerChannel
														,float fGlobalWordsPerChannel)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold,nLBSPThresholdOffset)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_fLocalWordsPerChannel(fLocalWordsPerChannel)
		,m_nLocalWords(0)
		,m_fGlobalWordsPerChannel(fGlobalWordsPerChannel)
		,m_nGlobalWords(0)
		,m_nMaxLocalDictionaries(0)
		,m_nFrameIndex(SIZE_MAX)
		,m_aapLocalDicts(nullptr)
		,m_apLocalWordList_1ch(nullptr)
		,m_apLocalWordList_3ch(nullptr)
		,m_apGlobalDict(nullptr)
		,m_apGlobalWordList_1ch(nullptr)
		,m_apGlobalWordList_3ch(nullptr)
		,m_apGlobalWordLookupTable(nullptr) {
	CV_Assert(m_fLocalWordsPerChannel>0.0f && m_fGlobalWordsPerChannel>0.0f);
	CV_Assert(m_nColorDistThreshold>0);
}

BackgroundSubtractorCBLBSP::~BackgroundSubtractorCBLBSP() {
	CleanupDictionaries();
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
	CleanupDictionaries();
	m_voKeyPoints = voNewKeyPoints;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nMaxLocalDictionaries = oInitImg.cols*oInitImg.rows;
	m_nLocalWords = ((size_t)(m_fLocalWordsPerChannel*m_nImgChannels))>0?(size_t)(m_fLocalWordsPerChannel*m_nImgChannels):1;
	m_nGlobalWords = ((size_t)(m_fGlobalWordsPerChannel*m_nImgChannels))>0?(size_t)(m_fGlobalWordsPerChannel*m_nImgChannels):1;
	m_nFrameIndex = 0;
	m_aapLocalDicts = new LocalWord*[m_nMaxLocalDictionaries*m_nLocalWords];
	memset(m_aapLocalDicts,0,sizeof(LocalWord*)*m_nMaxLocalDictionaries*m_nLocalWords);
#if USE_GLOBAL_WORDS
	m_apGlobalDict = new GlobalWord*[m_nGlobalWords];
	memset(m_apGlobalDict,0,sizeof(GlobalWord*)*m_nGlobalWords);
	m_apGlobalWordLookupTable = new GlobalWord*[m_nMaxLocalDictionaries];
	memset(m_apGlobalWordLookupTable,0,sizeof(GlobalWord*)*m_nMaxLocalDictionaries);
#endif //USE_GLOBAL_WORDS
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(5.0f);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(BGSCBLBSP_T_LOWER);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.3f);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.3f);
	m_oMeanSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanSegmResFrame = cv::Scalar(0.1f);
	m_oTempFGMask.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask = cv::Scalar_<uchar>(0);
	m_oTempFGMask2.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask2 = cv::Scalar_<uchar>(0);
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
	m_oFGMask_last_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated_inverted = cv::Scalar_<uchar>(0);
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
		const size_t nColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
		const size_t nDescDistThreshold = m_nDescDistThreshold;
		m_apLocalWordList_1ch = new LocalWord_1ch[nKeyPoints*m_nLocalWords];
		memset(m_apLocalWordList_1ch,0,sizeof(LocalWord_1ch)*nKeyPoints*m_nLocalWords);
		LocalWord_1ch* apLocalWordListIter = m_apLocalWordList_1ch;
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t nInitIters = (s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight);
			const size_t nInitWordOccIncr = std::max(LWORD_WEIGHT_OFFSET/nInitIters,(size_t)1);
			for(size_t n=0; n<nInitIters; ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oImgSize.width*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar nSampleColor = m_oLastColorFrame.data[idx_sample_color];
				const ushort nSampleIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				const uchar nSampleIntraDescBITS = popcount_ushort_8bitsLUT(nSampleIntraDesc);
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_1ch* pCurrLocalWord = ((LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]);
					if(pCurrLocalWord
							&& absdiff_uchar(nSampleColor,pCurrLocalWord->nColor)<=nColorDistThreshold
							&& ((pCurrLocalWord->nDescBITS<s_nDescDistTypeCutoff_1ch && nSampleIntraDescBITS<s_nDescDistTypeCutoff_1ch)?hdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc):(absdiff_uchar(nSampleIntraDescBITS,pCurrLocalWord->nDescBITS)+hdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc))/2)<=nDescDistThreshold) {
						pCurrLocalWord->nOccurrences += nInitWordOccIncr;
						break;
					}
				}
				if(nLocalWordIdx==m_nLocalWords) {
					nLocalWordIdx = m_nLocalWords-1;
					LocalWord_1ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx])
						pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					else {
						pCurrLocalWord = apLocalWordListIter++;
						m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
					}
					pCurrLocalWord->nColor = nSampleColor;
					pCurrLocalWord->nDesc = nSampleIntraDesc;
					pCurrLocalWord->nDescBITS = nSampleIntraDescBITS;
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET/4;
				}
				while(nLocalWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1] || m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]->nOccurrences>m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]->nOccurrences)) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]);
					--nLocalWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nLocalWordIdx=1; nLocalWordIdx<m_nLocalWords; ++nLocalWordIdx) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = apLocalWordListIter++;
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(nColorDistThreshold+1))-(int)nColorDistThreshold/2;
					pCurrLocalWord->nColor = cv::saturate_cast<uchar>((int)pRefLocalWord->nColor+nRandColorOffset);
					pCurrLocalWord->nDesc = pRefLocalWord->nDesc;
					pCurrLocalWord->nDescBITS = pRefLocalWord->nDescBITS;
					pCurrLocalWord->nOccurrences = (size_t)(pRefLocalWord->nOccurrences*((float)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords));
					m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_1ch==(apLocalWordListIter-nKeyPoints*m_nLocalWords));
#if USE_GLOBAL_WORDS
		m_apGlobalWordList_1ch = new GlobalWord_1ch[m_nGlobalWords];
		GlobalWord_1ch* apGlobalWordListIter = m_apGlobalWordList_1ch;
		size_t nGlobalWordFillIdx = 0;
		cv::Mat oGlobalDictPresenceLookupMap(m_oImgSize,CV_8UC1);
		oGlobalDictPresenceLookupMap = cv::Scalar_<uchar>(0);
		size_t nLocalDictIterIncr = (nKeyPoints/m_nGlobalWords)>0?(nKeyPoints/m_nGlobalWords):1;
		for(size_t k=0; k<nKeyPoints; k+=nLocalDictIterIncr) { // <=(m_nGlobalWords) gwords from (m_nGlobalWords) equally spaced keypoints
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_float = idx_orig_uchar*4;
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			const LocalWord_1ch* pRefBestLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict];
			const float fRefBestLocalWordWeight = GetLocalWordWeight(pRefBestLocalWord,1);
			size_t nGlobalWordIdx = 0;
			GlobalWord_1ch* pCurrGlobalWord;
			while(nGlobalWordIdx<nGlobalWordFillIdx) {
				pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
				if(absdiff_uchar(pCurrGlobalWord->nColor,pRefBestLocalWord->nColor)<=nColorDistThreshold && absdiff_uchar(pRefBestLocalWord->nDescBITS,pCurrGlobalWord->nDescBITS)<=nDescDistThreshold)
					break;
				++nGlobalWordIdx;
			}
			if(nGlobalWordIdx==nGlobalWordFillIdx) {
				pCurrGlobalWord = apGlobalWordListIter++;
				pCurrGlobalWord->nColor = pRefBestLocalWord->nColor;
				pCurrGlobalWord->nDescBITS = pRefBestLocalWord->nDescBITS;
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				++nGlobalWordFillIdx;
			}
			m_apGlobalWordLookupTable[idx_orig_uchar] = pCurrGlobalWord;
			float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_float);
			if((*pfCurrGlobalWordLocalWeight)<fRefBestLocalWordWeight) {
				pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
				(*pfCurrGlobalWordLocalWeight) += fRefBestLocalWordWeight;
			}
			oGlobalDictPresenceLookupMap.data[idx_orig_uchar] = UCHAR_MAX;
		}
		size_t nLocalDictWordIdxOffset = 0;
		size_t nLookupMapIdxOffset = (nLocalDictIterIncr/2>0)?(nLocalDictIterIncr/2):1;
		while(nGlobalWordFillIdx<m_nGlobalWords) {
			if(nLocalDictWordIdxOffset<m_nLocalWords) {
				size_t nLookupMapIdx = 0;
				while(nLookupMapIdx<nKeyPoints && nGlobalWordFillIdx<m_nGlobalWords) {
					if(m_aapLocalDicts[nLookupMapIdx*m_nLocalWords] && oGlobalDictPresenceLookupMap.data[nLookupMapIdx]<UCHAR_MAX) {
						const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[nLookupMapIdx*m_nLocalWords+nLocalDictWordIdxOffset];
						const float fRefLocalWordWeight = GetLocalWordWeight(pRefLocalWord,1);
						size_t nGlobalWordIdx = 0;
						GlobalWord_1ch* pCurrGlobalWord;
						while(nGlobalWordIdx<nGlobalWordFillIdx) {
							pCurrGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							if(absdiff_uchar(pCurrGlobalWord->nColor,pRefLocalWord->nColor)<=nColorDistThreshold && absdiff_uchar(pRefLocalWord->nDescBITS,pCurrGlobalWord->nDescBITS)<=nDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==nGlobalWordFillIdx) {
							pCurrGlobalWord = apGlobalWordListIter++;
							pCurrGlobalWord->nColor = pRefLocalWord->nColor;
							pCurrGlobalWord->nDescBITS = pRefLocalWord->nDescBITS;
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
							m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							++nGlobalWordFillIdx;
						}
						m_apGlobalWordLookupTable[nLookupMapIdx] = pCurrGlobalWord;
						float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4));
						if((*pfCurrGlobalWordLocalWeight)<fRefLocalWordWeight) {
							pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
							(*pfCurrGlobalWordLocalWeight) += fRefLocalWordWeight;
						}
						oGlobalDictPresenceLookupMap.data[nLookupMapIdx] = UCHAR_MAX;
					}
					nLookupMapIdx += nLookupMapIdxOffset;
				}
				nLookupMapIdxOffset = (nLookupMapIdxOffset/2>0)?(nLookupMapIdxOffset/2):1;
				++nLocalDictWordIdxOffset;
			}
			else {
				while(nGlobalWordFillIdx<m_nGlobalWords) {
					GlobalWord_1ch* pCurrGlobalWord = apGlobalWordListIter++;
					pCurrGlobalWord->nColor = rand()%(UCHAR_MAX+1);
					pCurrGlobalWord->nDescBITS = 0;
					pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
					pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
					pCurrGlobalWord->fLatestWeight = 0.0f;
					m_apGlobalDict[nGlobalWordFillIdx] = pCurrGlobalWord;
					++nGlobalWordFillIdx;
				}
				break;
			}
		}
		CV_Assert(nGlobalWordFillIdx==m_nGlobalWords && m_apGlobalWordList_1ch==(apGlobalWordListIter-m_nGlobalWords));
#endif //USE_GLOBAL_WORDS
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
				const uchar nCurrBGInitColor = oInitImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
		const size_t nTotColorDistThreshold = m_nColorDistThreshold*3;
		const size_t nTotDescDistThreshold = m_nDescDistThreshold*3;
		m_apLocalWordList_3ch = new LocalWord_3ch[nKeyPoints*m_nLocalWords];
		memset(m_apLocalWordList_3ch,0,sizeof(LocalWord_3ch)*nKeyPoints*m_nLocalWords);
		LocalWord_3ch* apLocalWordListIter = m_apLocalWordList_3ch;
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t nInitIters = (s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight);
			const size_t nInitWordOccIncr = std::max(LWORD_WEIGHT_OFFSET/nInitIters,(size_t)1);
			for(size_t n=0; n<nInitIters; ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = (m_oImgSize.width*y_sample + x_sample)*3;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar* const anSampleColor = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const anSampleIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				const uchar anSampleIntraDescBITS[3] = {popcount_ushort_8bitsLUT(anSampleIntraDesc[0]),popcount_ushort_8bitsLUT(anSampleIntraDesc[1]),popcount_ushort_8bitsLUT(anSampleIntraDesc[2])};
				const uchar nSampleIntraDescBITS = anSampleIntraDescBITS[0]+anSampleIntraDescBITS[1]+anSampleIntraDescBITS[2];
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					if(pCurrLocalWord
							&& L1dist_uchar(anSampleColor,pCurrLocalWord->anColor)<=nTotColorDistThreshold
							&& ((pCurrLocalWord->nDescBITS<s_nDescDistTypeCutoff_3ch && nSampleIntraDescBITS<s_nDescDistTypeCutoff_3ch)?hdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc):(absdiff_uchar(nSampleIntraDescBITS,pCurrLocalWord->nDescBITS)+hdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc))/2)<=nTotDescDistThreshold) {
						pCurrLocalWord->nOccurrences += nInitWordOccIncr;
						break;
					}
				}
				if(nLocalWordIdx==m_nLocalWords) {
					nLocalWordIdx = m_nLocalWords-1;
					LocalWord_3ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx])
						pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					else {
						pCurrLocalWord = apLocalWordListIter++;
						m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
					}
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = anSampleColor[c];
						pCurrLocalWord->anDesc[c] = anSampleIntraDesc[c];
						pCurrLocalWord->anDescBITS[c] = anSampleIntraDescBITS[c];
					}
					pCurrLocalWord->nDescBITS = nSampleIntraDescBITS;
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET/4;
				}
				while(nLocalWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1] || m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]->nOccurrences>m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]->nOccurrences)) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx],m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx-1]);
					--nLocalWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nLocalWordIdx=1; nLocalWordIdx<m_nLocalWords; ++nLocalWordIdx) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = apLocalWordListIter++;
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(m_nColorDistThreshold+1))-(int)m_nColorDistThreshold/2;
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = cv::saturate_cast<uchar>((int)pRefLocalWord->anColor[c]+nRandColorOffset);
						pCurrLocalWord->anDesc[c] = pRefLocalWord->anDesc[c];
						pCurrLocalWord->anDescBITS[c] = pRefLocalWord->anDescBITS[c];
					}
					pCurrLocalWord->nDescBITS = pRefLocalWord->nDescBITS;
					pCurrLocalWord->nOccurrences = (size_t)(pRefLocalWord->nOccurrences*((float)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords));
					m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_3ch==(apLocalWordListIter-nKeyPoints*m_nLocalWords));
#if USE_GLOBAL_WORDS
		m_apGlobalWordList_3ch = new GlobalWord_3ch[m_nGlobalWords];
		GlobalWord_3ch* apGlobalWordListIter = m_apGlobalWordList_3ch;
		size_t nGlobalWordFillIdx = 0;
		cv::Mat oGlobalDictPresenceLookupMap(m_oImgSize,CV_8UC1);
		oGlobalDictPresenceLookupMap = cv::Scalar_<uchar>(0);
		size_t nLocalDictIterIncr = (nKeyPoints/m_nGlobalWords)>0?(nKeyPoints/m_nGlobalWords):1;
		for(size_t k=0; k<nKeyPoints; k+=nLocalDictIterIncr) { // <=(m_nGlobalWords) gwords from (m_nGlobalWords) equally spaced keypoints
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			const size_t idx_orig_float = idx_orig_uchar*4;
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			const LocalWord_3ch* pRefBestLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict];
			const float fRefBestLocalWordWeight = GetLocalWordWeight(pRefBestLocalWord,1);
			size_t nGlobalWordIdx = 0;
			GlobalWord_3ch* pCurrGlobalWord;
			while(nGlobalWordIdx<nGlobalWordFillIdx) {
				pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
				if(L1dist_uchar(pCurrGlobalWord->anColor,pRefBestLocalWord->anColor)<=nTotColorDistThreshold && absdiff_uchar(pRefBestLocalWord->nDescBITS,pCurrGlobalWord->nDescBITS)<=nTotDescDistThreshold)
					break;
				++nGlobalWordIdx;
			}
			if(nGlobalWordIdx==nGlobalWordFillIdx) {
				pCurrGlobalWord = apGlobalWordListIter++;
				for(size_t c=0; c<3; ++c)
					pCurrGlobalWord->anColor[c] = pRefBestLocalWord->anColor[c];
				pCurrGlobalWord->nDescBITS = pRefBestLocalWord->nDescBITS;
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				++nGlobalWordFillIdx;
			}
			m_apGlobalWordLookupTable[idx_orig_uchar] = pCurrGlobalWord;
			float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_float);
			if((*pfCurrGlobalWordLocalWeight)<fRefBestLocalWordWeight) {
				pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
				(*pfCurrGlobalWordLocalWeight) += fRefBestLocalWordWeight;
			}
			oGlobalDictPresenceLookupMap.data[idx_orig_uchar] = UCHAR_MAX;
		}
		size_t nLocalDictWordIdxOffset = 0;
		size_t nLookupMapIdxOffset = (nLocalDictIterIncr/2>0)?(nLocalDictIterIncr/2):1;
		while(nGlobalWordFillIdx<m_nGlobalWords) {
			if(nLocalDictWordIdxOffset<m_nLocalWords) {
				size_t nLookupMapIdx = 0;
				while(nLookupMapIdx<nKeyPoints && nGlobalWordFillIdx<m_nGlobalWords) {
					if(m_aapLocalDicts[nLookupMapIdx*m_nLocalWords] && oGlobalDictPresenceLookupMap.data[nLookupMapIdx]<UCHAR_MAX) {
						const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[nLookupMapIdx*m_nLocalWords+nLocalDictWordIdxOffset];
						const float fRefLocalWordWeight = GetLocalWordWeight(pRefLocalWord,1);
						size_t nGlobalWordIdx = 0;
						GlobalWord_3ch* pCurrGlobalWord;
						while(nGlobalWordIdx<nGlobalWordFillIdx) {
							pCurrGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							if(L1dist_uchar(pCurrGlobalWord->anColor,pRefLocalWord->anColor)<=nTotColorDistThreshold && absdiff_uchar(pRefLocalWord->nDescBITS,pCurrGlobalWord->nDescBITS)<=nTotDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==nGlobalWordFillIdx) {
							pCurrGlobalWord = apGlobalWordListIter++;
							for(size_t c=0; c<3; ++c)
								pCurrGlobalWord->anColor[c] = pRefLocalWord->anColor[c];
							pCurrGlobalWord->nDescBITS = pRefLocalWord->nDescBITS;
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
							m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							++nGlobalWordFillIdx;
						}
						m_apGlobalWordLookupTable[nLookupMapIdx] = pCurrGlobalWord;
						float* pfCurrGlobalWordLocalWeight = (float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4));
						if((*pfCurrGlobalWordLocalWeight)<fRefLocalWordWeight) {
							pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
							(*pfCurrGlobalWordLocalWeight) += fRefLocalWordWeight;
						}
						oGlobalDictPresenceLookupMap.data[nLookupMapIdx] = UCHAR_MAX;
					}
					nLookupMapIdx += nLookupMapIdxOffset;
				}
				nLookupMapIdxOffset = (nLookupMapIdxOffset/2>0)?(nLookupMapIdxOffset/2):1;
				++nLocalDictWordIdxOffset;
			}
			else {
				while(nGlobalWordFillIdx<m_nGlobalWords) {
					GlobalWord_3ch* pCurrGlobalWord = apGlobalWordListIter++;
					for(size_t c=0; c<3; ++c)
						pCurrGlobalWord->anColor[c] = rand()%(UCHAR_MAX+1);
					pCurrGlobalWord->nDescBITS = 0;
					pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
					pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
					pCurrGlobalWord->fLatestWeight = 0.0f;
					m_apGlobalDict[nGlobalWordFillIdx] = pCurrGlobalWord;
					++nGlobalWordFillIdx;
				}
				break;
			}
		}
		CV_Assert(nGlobalWordFillIdx==m_nGlobalWords && m_apGlobalWordList_3ch==(apGlobalWordListIter-m_nGlobalWords));
#endif //USE_GLOBAL_WORDS
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
	const size_t nKeyPoints = m_voKeyPoints.size();
	const size_t nCurrGlobalWordUpdateRate = /*learningRateOverride>0?(size_t)ceil(learningRateOverride):*/GWORD_REPRESENTATION_UPDATE_RATE;
	const size_t nWordOccIncr = m_nFrameIndex<100?((LWORD_WEIGHT_OFFSET*2)/100):1;
	const size_t nSamplesForMean = std::min(++m_nFrameIndex,(size_t)BGSCBLBSP_N_SAMPLES_FOR_MEAN);
#if DISPLAY_CBLBSP_DEBUG_INFO
	std::vector<std::string> vsWordModList(m_nMaxLocalDictionaries*m_nLocalWords);
	uchar anDBGColor[3] = {0,0,0};
	ushort anDBGIntraDesc[3] = {0,0,0};
	bool bDBGMaskResult = false;
	bool bDBGMaskModifiedByGDict = false;
	GlobalWord* pDBGGlobalWordModifier = nullptr;
	float fDBGGlobalWordModifierLocalWeight = 0.0f;
	size_t idx_dbg_ldict = UINT_MAX;
	size_t nDBGWordOccIncr = nWordOccIncr;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
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
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			const float fCurrDistFactor = (*pfCurrMeanMinDist+*pfCurrMeanLastDist)/2;
			const float fLocalWordsWeightSumThreshold = 1.0f;///sqrt(*pfCurrDistThresholdFactor+(1.0f-BGSCBLBSP_R_LOWER));
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):LWORD_REPRESENTATION_UPDATE_RATE;//(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			const uchar nCurrIntraDescBITS = popcount_ushort_8bitsLUT(nCurrIntraDesc);
			const size_t nCurrWordOccIncr = (
												((*pfCurrMeanSegmRes)>BGSCBLBSP_GHOST_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_D_MAX2) ||
												((*pfCurrMeanSegmRes)>BGSCBLBSP_GHOST_DETECTION_S_MIN && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_D_MAX)
												//((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
												//((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)
											) ? std::max((size_t)16,nWordOccIncr) : nWordOccIncr;
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			float fMaxLocalWordWeight = 0.0f;
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,pCurrLocalWord->nColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					size_t nIntraDescDist, nInterDescDist;
					if(pCurrLocalWord->nDescBITS<s_nDescDistTypeCutoff_1ch && nCurrIntraDescBITS<s_nDescDistTypeCutoff_1ch) {
						nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,pCurrLocalWord->nDesc);
						if(nIntraDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
						LBSP::computeGrayscaleDescriptor(oInputImg,pCurrLocalWord->nColor,x,y,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->nColor],nCurrInterDesc);
						nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,pCurrLocalWord->nDesc);
						if(nInterDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
					}
					else {
						nIntraDescDist = (absdiff_uchar(nCurrIntraDescBITS,pCurrLocalWord->nDescBITS)+hdist_ushort_8bitLUT(nCurrIntraDesc,pCurrLocalWord->nDesc))/2;
						if(nIntraDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
						LBSP::computeGrayscaleDescriptor(oInputImg,pCurrLocalWord->nColor,x,y,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->nColor],nCurrInterDesc);
						nInterDescDist = (absdiff_uchar(popcount_ushort_8bitsLUT(nCurrInterDesc),pCurrLocalWord->nDescBITS)+hdist_ushort_8bitLUT(nCurrInterDesc,pCurrLocalWord->nDesc))/2;
						if(nInterDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
					}
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					const size_t nSumDist = std::min((size_t)(OVERLOAD_GRAD_PROP*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
#if DISPLAY_CBLBSP_DEBUG_INFO
					vsWordModList[idx_ldict+nLocalWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					pCurrLocalWord->nOccurrences += nCurrWordOccIncr;
					const float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					fPotentialLocalWordsWeightSum += fCurrLocalWordWeight;
					fMaxLocalWordWeight = std::max(fMaxLocalWordWeight,fCurrLocalWordWeight);
					if(!m_oFGMask_last.data[idx_uchar] && nDescDist<=nCurrDescDistThreshold/2 && (rand()%nCurrLocalWordUpdateRate)==0) { // @@@@ using (intra+inter)/2...
						pCurrLocalWord->nColor = nCurrColor;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
					if(nMinColorDist>nColorDist)
						nMinColorDist = nColorDist;
					if(nMinDescDist>nDescDist)
						nMinDescDist = nDescDist;
					if(nMinSumDist>nSumDist)
						nMinSumDist = nSumDist;
				}
				failedcheck1ch:
				if(nLocalWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				++nLocalWordIdx;
			}
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2)/nSamplesForMean;
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinSumDist/s_nColorMaxDataRange_1ch)/nSamplesForMean;
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinColorDist/s_nColorMaxDataRange_1ch)/nSamplesForMean;
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(nSamplesForMean-1) + ((float)(absdiff_uchar(nLastColor,nCurrColor))/s_nColorMaxDataRange_1ch+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/s_nDescMaxDataRange_1ch)/2)/nSamplesForMean; // @@@@ add bit trick?
			//*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(nSamplesForMean-1) + (float)(absdiff_uchar(nLastColor,nCurrColor))/s_nColorMaxDataRange_1ch)/nSamplesForMean;
			if(fPotentialLocalWordsWeightSum>=fLocalWordsWeightSumThreshold) {
				// == background
				*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinSumDist/s_nColorMaxDataRange_1ch)/nSamplesForMean;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(nSamplesForMean-1))/nSamplesForMean;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					//*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
					//*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanLastDist);
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/fCurrDistFactor;
					if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
						*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
				const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
				if(m_aapLocalDicts[idx_rand_ldict]) {
					const size_t idx_rand_flt32 = idx_rand_uchar*4;
					const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
					const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+idx_rand_flt32));
					//const float fRandLearningRate = *((float*)(m_oUpdateRateFrame.data+idx_rand_flt32));
					const size_t n_rand = rand();
					//const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(fRandLearningRate);
					const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
					if((n_rand%nCurrLocalWordNeighborSpreadRate)==0 || ((n_rand%4)==0 && (
						//(fRandMeanSegmRes>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && fRandMeanLastDist>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
						//(fRandMeanSegmRes>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && fRandMeanLastDist>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2) ||
						(fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX) ||
						(fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX)))) {
						size_t nRandLocalWordIdx = 0;
						float fPotentialRandLocalWordsWeightSum = 0.0f;
						while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
							LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							if(absdiff_uchar(nCurrColor,pRandLocalWord->nColor)<=nCurrColorDistThreshold && ((pRandLocalWord->nDescBITS<s_nDescDistTypeCutoff_1ch && nCurrIntraDescBITS<s_nDescDistTypeCutoff_1ch)?hdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc):(absdiff_uchar(nCurrIntraDescBITS,pRandLocalWord->nDescBITS)+hdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc))/2)<=nCurrDescDistThreshold) {
								pRandLocalWord->nOccurrences += (size_t)std::max((LWORD_WEIGHT_OFFSET-(int)pRandLocalWord->nOccurrences)/8,(int)nCurrWordOccIncr);
								fPotentialRandLocalWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
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
							pRandLocalWord->nDescBITS = nCurrIntraDescBITS;
							pRandLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET/8;
							pRandLocalWord->nFirstOcc = m_nFrameIndex;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
							vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
						}
					}
				}
#if USE_GLOBAL_WORDS
				if((rand()%nCurrGlobalWordUpdateRate)==0) {
					GlobalWord_1ch* pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalWordLookupTable[idx_uchar];
					if(!pLastMatchedGlobalWord || absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold || absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold) {
						size_t nGlobalWordIdx = 0;
						while(nGlobalWordIdx<m_nGlobalWords) {
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold && absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							pLastMatchedGlobalWord->nColor = nCurrColor;
							pLastMatchedGlobalWord->nDescBITS = nCurrIntraDescBITS;
							pLastMatchedGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pLastMatchedGlobalWord->fLatestWeight = 0.0f;
						}
						m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
					}
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
#endif //USE_GLOBAL_WORDS
			}
			else {
				// == foreground
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinSumDist/s_nColorMaxDataRange_1ch + (fLocalWordsWeightSumThreshold-fMaxLocalWordWeight))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinSumDist/s_nColorMaxDataRange_1ch+(fLocalWordsWeightSumThreshold-fMaxLocalWordWeight),1.0f))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinSumDist/s_nColorMaxDataRange_1ch+(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum)/4,1.0f))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)pow(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum,2)/4,1.0f))/nSamplesForMean;
				*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + 1.0f)/nSamplesForMean;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(nSamplesForMean-1) + 1.0f)/nSamplesForMean;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					//*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanLastDist);
					//*pfCurrLearningRate += BGSCBLBSP_T_INCR/fCurrDistFactor;
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
				GlobalWord_1ch* pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalWordLookupTable[idx_uchar];
				if(!pLastMatchedGlobalWord || absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold || absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold) {
					size_t nGlobalWordIdx = 0;
					while(nGlobalWordIdx<m_nGlobalWords) {
						pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
						if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold && absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold)
							break;
						++nGlobalWordIdx;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || (*(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32))/2+fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
					if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						const size_t nNewLocalWordIdx = m_nLocalWords-1;
						LocalWord_1ch* pNewLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
						pNewLocalWord->nColor = nCurrColor;
						pNewLocalWord->nDesc = nCurrIntraDesc;
						pNewLocalWord->nDescBITS = nCurrIntraDescBITS;
						pNewLocalWord->nOccurrences = nCurrWordOccIncr;
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
#if DISPLAY_CBLBSP_DEBUG_INFO
				else if(y==nDebugCoordY && x==nDebugCoordX) {
					bDBGMaskModifiedByGDict = true;
					pDBGGlobalWordModifier = pLastMatchedGlobalWord;
					fDBGGlobalWordModifierLocalWeight = *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
				}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
			}
			while(nLocalWordIdx<m_nLocalWords) {
				if(nLocalWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				++nLocalWordIdx;
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			//if( ((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
			//if( ((*pfCurrMeanLastDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
			/*if(y==nDebugCoordY && x==nDebugCoordX) {
				printf("[fCurrDistFactor:%f]>[BGSCBLBSP_R2_OFFST:%f] && [m_oBlinksFrame.data[idx_uchar]:%d]>0\n",fCurrDistFactor,BGSCBLBSP_R2_OFFST,(int)m_oBlinksFrame.data[idx_uchar]);
				printf("[(*pfCurrMeanSegmRes):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN:%f] && [(*pfCurrMeanLastDist):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN:%f]\n",(*pfCurrMeanSegmRes),BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN,(*pfCurrMeanLastDist),BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN);
				printf("[(*pfCurrMeanSegmRes):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2:%f] && [(*pfCurrMeanLastDist):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2:%f]\n",(*pfCurrMeanSegmRes),BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2,(*pfCurrMeanLastDist),BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2);
			}*/
			if( (fCurrDistFactor>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && fCurrDistFactor>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && fCurrDistFactor>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else if((*pfCurrDistThresholdVariationFactor)>0) {
				//(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
				(*pfCurrDistThresholdVariationFactor) -= (m_oFGMask_last.data[idx_uchar]>0)?BGSCBLBSP_R2_DECR/4:BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<0)
					(*pfCurrDistThresholdVariationFactor) = 0;
			}
			//if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSCBLBSP_R_SCALE) {
			//if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanLastDist)*BGSCBLBSP_R_SCALE) {
			if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+fCurrDistFactor*BGSCBLBSP_R_SCALE) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_LOWER) {
				if((*pfCurrDistThresholdVariationFactor)>BGSCBLBSP_R2_DECR)
					(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				else
					(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/BGSCBLBSP_R2_DECR;
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
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
				nDBGWordOccIncr = std::max(nDBGWordOccIncr,nCurrWordOccIncr);
			}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<nKeyPoints; ++k) {
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
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			const float fCurrDistFactor = (*pfCurrMeanMinDist+*pfCurrMeanLastDist)/2;
			const float fLocalWordsWeightSumThreshold = 1.0f;///sqrt(*pfCurrDistThresholdFactor+(1.0f-BGSCBLBSP_R_LOWER));
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):LWORD_REPRESENTATION_UPDATE_RATE;//(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrTotColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const size_t nCurrTotDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
			const size_t nCurrSCColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const size_t nCurrSCDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			const uchar anCurrIntraDescBITS[3] = {popcount_ushort_8bitsLUT(anCurrIntraDesc[0]),popcount_ushort_8bitsLUT(anCurrIntraDesc[1]),popcount_ushort_8bitsLUT(anCurrIntraDesc[2])};
			const uchar nCurrIntraDescBITS = anCurrIntraDescBITS[0]+anCurrIntraDescBITS[1]+anCurrIntraDescBITS[2];
			const size_t nCurrWordOccIncr = (
												((*pfCurrMeanSegmRes)>BGSCBLBSP_GHOST_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_D_MAX2) ||
												((*pfCurrMeanSegmRes)>BGSCBLBSP_GHOST_DETECTION_S_MIN && (*pfCurrMeanLastDist)<BGSCBLBSP_GHOST_DETECTION_D_MAX)
												//((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
												//((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)
											) ? std::max((size_t)16,nWordOccIncr) : nWordOccIncr;
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			float fMaxLocalWordWeight = 0.0f;
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				size_t nTotColorDist = 0;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],pCurrLocalWord->anColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					size_t nIntraDescDist, nInterDescDist;
					if(pCurrLocalWord->nDescBITS<s_nDescDistTypeCutoff_3ch && nCurrIntraDescBITS<s_nDescDistTypeCutoff_3ch) {
						nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],pCurrLocalWord->anDesc[c]);
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,pCurrLocalWord->anColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[c]],anCurrInterDesc[c]);
						nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],pCurrLocalWord->anDesc[c]);
						if(nInterDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
					}
					else {
						nIntraDescDist = (absdiff_uchar(anCurrIntraDescBITS[c],pCurrLocalWord->anDescBITS[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],pCurrLocalWord->anDesc[c]))/2;
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,pCurrLocalWord->anColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[pCurrLocalWord->anColor[c]],anCurrInterDesc[c]);
						nInterDescDist = (absdiff_uchar(popcount_ushort_8bitsLUT(anCurrInterDesc[c]),pCurrLocalWord->anDescBITS[c])+hdist_ushort_8bitLUT(anCurrInterDesc[c],pCurrLocalWord->anDesc[c]))/2;
						if(nInterDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
					}
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					const size_t nSumDist = std::min((size_t)(OVERLOAD_GRAD_PROP*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(nTotDescDist<=nCurrTotDescDistThreshold && nTotSumDist<nCurrTotColorDistThreshold) {
#if DISPLAY_CBLBSP_DEBUG_INFO
					vsWordModList[idx_ldict+nLocalWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					pCurrLocalWord->nOccurrences += nCurrWordOccIncr;
					const float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					fPotentialLocalWordsWeightSum += fCurrLocalWordWeight;
					fMaxLocalWordWeight = std::max(fMaxLocalWordWeight,fCurrLocalWordWeight);
					if(!m_oFGMask_last.data[idx_uchar] && nTotDescDist<=nCurrTotDescDistThreshold/2 && (rand()%nCurrLocalWordUpdateRate)==0) { // @@@@ using (intra+inter)/2...
						for(size_t c=0; c<3; ++c)
							pCurrLocalWord->anColor[c] = anCurrColor[c];
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
					if(nMinTotColorDist>nTotColorDist)
						nMinTotColorDist = nTotColorDist;
					if(nMinTotDescDist>nTotDescDist)
						nMinTotDescDist = nTotDescDist;
					if(nMinTotSumDist>nTotSumDist)
						nMinTotSumDist = nTotSumDist;
				}
				failedcheck3ch:
				if(nLocalWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				++nLocalWordIdx;
			}
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2)/nSamplesForMean;
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinTotSumDist/s_nColorMaxDataRange_3ch)/nSamplesForMean;
			//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinTotColorDist/s_nColorMaxDataRange_3ch)/nSamplesForMean;
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(nSamplesForMean-1) + ((float)(L1dist_uchar(anLastColor,anCurrColor))/s_nColorMaxDataRange_3ch+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/s_nDescMaxDataRange_3ch)/2)/nSamplesForMean; // @@@@ add bit trick?
			//*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(nSamplesForMean-1) + (float)(L1dist_uchar(anLastColor,anCurrColor))/s_nColorMaxDataRange_3ch)/nSamplesForMean;
			if(fPotentialLocalWordsWeightSum>=fLocalWordsWeightSumThreshold) {
				// == background
				*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinTotSumDist/s_nColorMaxDataRange_3ch)/nSamplesForMean;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(nSamplesForMean-1))/nSamplesForMean;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					//*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
					//*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanLastDist);
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/fCurrDistFactor;
					if((*pfCurrLearningRate)<BGSCBLBSP_T_LOWER)
						*pfCurrLearningRate = BGSCBLBSP_T_LOWER;
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
				const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
				if(m_aapLocalDicts[idx_rand_ldict]) {
					const size_t idx_rand_flt32 = idx_rand_uchar*4;
					const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
					const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+idx_rand_flt32));
					//const float fRandLearningRate = *((float*)(m_oUpdateRateFrame.data+idx_rand_flt32));
					const size_t n_rand = rand();
					//const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(fRandLearningRate);
					const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
					if((n_rand%nCurrLocalWordNeighborSpreadRate)==0 || ((n_rand%4)==0 && (
						//(fRandMeanSegmRes>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && fRandMeanLastDist>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
						//(fRandMeanSegmRes>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && fRandMeanLastDist>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2) ||
						(fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX) ||
						(fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX)))) {
						size_t nRandLocalWordIdx = 0;
						float fPotentialRandLocalWordsWeightSum = 0.0f;
						while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
							LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							if(L1dist_uchar(anCurrColor,pRandLocalWord->anColor)<=nCurrTotColorDistThreshold && ((pRandLocalWord->nDescBITS<s_nDescDistTypeCutoff_3ch && nCurrIntraDescBITS<s_nDescDistTypeCutoff_3ch)?hdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc):(absdiff_uchar(nCurrIntraDescBITS,pRandLocalWord->nDescBITS)+hdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc))/2)<=nCurrTotDescDistThreshold) {
								pRandLocalWord->nOccurrences += (size_t)std::max((LWORD_WEIGHT_OFFSET-(int)pRandLocalWord->nOccurrences)/8,(int)nCurrWordOccIncr);
								fPotentialRandLocalWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
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
								pRandLocalWord->anDescBITS[c] = anCurrIntraDescBITS[c];
							}
							pRandLocalWord->nDescBITS = nCurrIntraDescBITS;
							pRandLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET/8;
							pRandLocalWord->nFirstOcc = m_nFrameIndex;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
							vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
						}
					}
				}
#if USE_GLOBAL_WORDS
				if((rand()%nCurrGlobalWordUpdateRate)==0) {
					GlobalWord_3ch* pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalWordLookupTable[idx_uchar];
					if(!pLastMatchedGlobalWord || L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)>nCurrTotColorDistThreshold || absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold) {
						size_t nGlobalWordIdx = 0;
						while(nGlobalWordIdx<m_nGlobalWords) {
							pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							if(L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)<=nCurrTotColorDistThreshold && absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold)
								break;
							++nGlobalWordIdx;
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
						m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
					}
					float* pfLastMatchedGlobalWord_LocalWeight = (float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
					if((*pfLastMatchedGlobalWord_LocalWeight)<fPotentialLocalWordsWeightSum) {
						pLastMatchedGlobalWord->fLatestWeight += fPotentialLocalWordsWeightSum;
						*pfLastMatchedGlobalWord_LocalWeight += fPotentialLocalWordsWeightSum;
					}
				}
#endif //USE_GLOBAL_WORDS
			}
			else {
				// == foreground
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + (float)nMinTotSumDist/s_nColorMaxDataRange_3ch + (fLocalWordsWeightSumThreshold-fMaxLocalWordWeight))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(fLocalWordsWeightSumThreshold-fMaxLocalWordWeight),1.0f))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum)/4,1.0f))/nSamplesForMean;
				//*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + std::min((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)pow(fLocalWordsWeightSumThreshold-fPotentialLocalWordsWeightSum,2)/4,1.0f))/nSamplesForMean;
				*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(nSamplesForMean-1) + 1.0f)/nSamplesForMean;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(nSamplesForMean-1) + 1.0f)/nSamplesForMean;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					//*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanLastDist);
					//*pfCurrLearningRate += BGSCBLBSP_T_INCR/fCurrDistFactor;
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
				GlobalWord_3ch* pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalWordLookupTable[idx_uchar];
				if(!pLastMatchedGlobalWord || L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)>nCurrTotColorDistThreshold || absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold) {
					size_t nGlobalWordIdx = 0;
					while(nGlobalWordIdx<m_nGlobalWords) {
						pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
						if(L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)<=nCurrTotColorDistThreshold && absdiff_uchar(nCurrIntraDescBITS,pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold)
							break;
						++nGlobalWordIdx;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || (*(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32))/2+fPotentialLocalWordsWeightSum<fLocalWordsWeightSumThreshold) {
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
					if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						const size_t nNewLocalWordIdx = m_nLocalWords-1;
						LocalWord_3ch* pNewLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
						for(size_t c=0; c<3; ++c) {
							pNewLocalWord->anColor[c] = anCurrColor[c];
							pNewLocalWord->anDesc[c] = anCurrIntraDesc[c];
							pNewLocalWord->anDescBITS[c] = anCurrIntraDescBITS[c];
						}
						pNewLocalWord->nDescBITS = nCurrIntraDescBITS;
						pNewLocalWord->nOccurrences = nCurrWordOccIncr;
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
#if DISPLAY_CBLBSP_DEBUG_INFO
				else if(y==nDebugCoordY && x==nDebugCoordX) {
					bDBGMaskModifiedByGDict = true;
					pDBGGlobalWordModifier = pLastMatchedGlobalWord;
					fDBGGlobalWordModifierLocalWeight = *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32);
				}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
			}
			while(nLocalWordIdx<m_nLocalWords) {
				if(nLocalWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nLocalWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nLocalWordIdx],m_aapLocalDicts[idx_ldict+nLocalWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_INFO
					std::swap(vsWordModList[idx_ldict+nLocalWordIdx],vsWordModList[idx_ldict+nLocalWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_INFO
				}
				++nLocalWordIdx;
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			//if( ((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
			//if( ((*pfCurrMeanLastDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
			/*if(y==nDebugCoordY && x==nDebugCoordX) {
				printf("[fCurrDistFactor:%f]>[BGSCBLBSP_R2_OFFST:%f] && [m_oBlinksFrame.data[idx_uchar]:%d]>0\n",fCurrDistFactor,BGSCBLBSP_R2_OFFST,(int)m_oBlinksFrame.data[idx_uchar]);
				printf("[(*pfCurrMeanSegmRes):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN:%f] && [(*pfCurrMeanLastDist):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN:%f]\n",(*pfCurrMeanSegmRes),BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN,(*pfCurrMeanLastDist),BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN);
				printf("[(*pfCurrMeanSegmRes):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2:%f] && [(*pfCurrMeanLastDist):%f]>[BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2:%f]\n",(*pfCurrMeanSegmRes),BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2,(*pfCurrMeanLastDist),BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2);
			}*/
			if( (fCurrDistFactor>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && fCurrDistFactor>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && fCurrDistFactor>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				if((*pfCurrDistThresholdVariationFactor)<BGSCBLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			}
			else if((*pfCurrDistThresholdVariationFactor)>0) {
				//(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
				(*pfCurrDistThresholdVariationFactor) -= (m_oFGMask_last.data[idx_uchar]>0)?BGSCBLBSP_R2_DECR/4:BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<0)
					(*pfCurrDistThresholdVariationFactor) = 0;
			}
			//if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSCBLBSP_R_SCALE) {
			//if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+(*pfCurrMeanLastDist)*BGSCBLBSP_R_SCALE) {
			if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER+fCurrDistFactor*BGSCBLBSP_R_SCALE) {
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSCBLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSCBLBSP_R_LOWER) {
				if((*pfCurrDistThresholdVariationFactor)>BGSCBLBSP_R2_DECR)
					(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				else
					(*pfCurrDistThresholdFactor) -= BGSCBLBSP_R_DECR/BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdFactor)<BGSCBLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSCBLBSP_R_LOWER;
			}
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
#if DISPLAY_CBLBSP_DEBUG_INFO
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = anCurrColor[c];
					anDBGIntraDesc[c] = anCurrIntraDesc[c];
				}
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
				nDBGWordOccIncr = std::max(nDBGWordOccIncr,nCurrWordOccIncr);
			}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		}
	}
	for(size_t nGlobalWordIdx=1; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx)
		if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)
			std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
	if(!(m_nFrameIndex%(nCurrGlobalWordUpdateRate))) {
		for(size_t nGlobalWordIdx=0; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx) {
			if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight==0.0f)
				continue;
			if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight<GWORD_WEIGHT_SUM_THRESHOLD) {
				m_apGlobalDict[nGlobalWordIdx]->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap = cv::Scalar(0.0f);
			}
			else {
				m_apGlobalDict[nGlobalWordIdx]->fLatestWeight *= GWORD_WEIGHT_DECIMATION_FACTOR;
				m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap *= GWORD_WEIGHT_DECIMATION_FACTOR;
				cv::blur(m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,cv::Size(7,7),cv::Point(-1,-1),cv::BORDER_REPLICATE);
				//cv::GaussianBlur(m_oDistThresholdVariationFrame,m_oDistThresholdVariationFrame,cv::Size(5,5),0);
			}
		}
	}
#if DISPLAY_CBLBSP_DEBUG_INFO
	if(idx_dbg_ldict!=UINT_MAX) {
		std::cout << std::endl;
		cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
		/*cv::Mat gwords_coverage(m_oImgSize,CV_32FC1);
		gwords_coverage = cv::Scalar(0.0f);
		for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords; ++nDBGWordIdx)
			cv::max(gwords_coverage,m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap,gwords_coverage);
		cv::imshow("gwords_coverage",gwords_coverage);
		std::string asDBGStrings[5] = {"gword[0]","gword[1]","gword[2]","gword[3]","gword[4]"};
		for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords && nDBGWordIdx<5; ++nDBGWordIdx)
			cv::imshow(asDBGStrings[nDBGWordIdx],m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap);
		double minVal,maxVal;
		cv::minMaxIdx(gwords_coverage,&minVal,&maxVal);
		std::cout << " " << m_nFrameIndex << " : gwords_coverage min=" << minVal << ", max=" << maxVal << std::endl;*/
		if(false) {
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
		cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
		cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,cv::Size(320,240));
		cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
		cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,cv::Size(320,240));
		cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oMeanSegmResFrameNormalized; m_oMeanSegmResFrame.copyTo(oMeanSegmResFrameNormalized);
		cv::circle(oMeanSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oMeanSegmResFrameNormalized,oMeanSegmResFrameNormalized,cv::Size(320,240));
		cv::imshow("s(x)",oMeanSegmResFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " s(" << dbgpt << ") = " << m_oMeanSegmResFrame.at<float>(dbgpt) << std::endl;
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
	}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
	cv::bitwise_xor(oCurrFGMask,m_oPureFGMask_last,m_oPureFGBlinkMask_curr);
	cv::bitwise_or(m_oPureFGBlinkMask_curr,m_oPureFGBlinkMask_last,m_oBlinksFrame);
	m_oPureFGBlinkMask_curr.copyTo(m_oPureFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oPureFGMask_last);
	cv::morphologyEx(oCurrFGMask,m_oTempFGMask,cv::MORPH_CLOSE,cv::Mat());
	cv::floodFill(m_oTempFGMask,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oTempFGMask,m_oTempFGMask);
	cv::bitwise_or(oCurrFGMask,m_oTempFGMask,oCurrFGMask);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_and(m_oPureFGMask_last,m_oFGMask_last_dilated_inverted,m_oTempFGMask2);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_and(m_oTempFGMask2,m_oFGMask_last_dilated_inverted,m_oTempFGMask2);
	cv::bitwise_or(m_oBlinksFrame,m_oTempFGMask2,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
}

void BackgroundSubtractorCBLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	// @@@@@@ TO BE REWRITTEN FOR WORD-BASED RECONSTRUCTION
	/*for(size_t w=0; w<m_nLocalWords; ++w) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_nimg = m_voBGColorSamples[w].step.p[0]*y + m_voBGColorSamples[w].step.p[1]*x;
				const size_t idx_flt32 = idx_nimg*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
				const uchar* const oBGImgPtr = m_voBGColorSamples[w].data+idx_nimg;
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nLocalWords;
			}
		}
	}*/
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
#if USE_GLOBAL_WORDS
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
	if(m_apGlobalWordLookupTable) {
		delete[] m_apGlobalWordLookupTable;
		m_apGlobalWordLookupTable = nullptr;
	}
#endif //USE_GLOBAL_WORDS
}

float BackgroundSubtractorCBLBSP::GetLocalWordWeight(const LocalWord* w, size_t nCurrFrame) {
	return (float)(w->nOccurrences)/((w->nLastOcc-w->nFirstOcc)/2+(nCurrFrame-w->nLastOcc)+LWORD_WEIGHT_OFFSET);
}

float BackgroundSubtractorCBLBSP::GetGlobalWordWeight(const GlobalWord* w) {
	return (float)cv::sum(w->oSpatioOccMap).val[0];
}

BackgroundSubtractorCBLBSP::LocalWord::~LocalWord() {}

BackgroundSubtractorCBLBSP::GlobalWord::~GlobalWord() {}
