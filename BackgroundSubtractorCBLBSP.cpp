#include "BackgroundSubtractorCBLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// @@@@@@ FOR FEEDBACK LOOPS, VARYING WEIGHT THRESHOLD > VARYING DIST THRESHOLD

// local define used for debug purposes only
#define DISPLAY_CBLBSP_DEBUG_INFO 0
// local define used to specify whether to use global words or not
#define USE_GLOBAL_WORDS 1
// local define for the gradient proportion value used in color+grad distance calculations
#define OVERLOAD_GRAD_PROP ((1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))*0.5f)
// local define for the lword representation update rate
#define LWORD_REPRESENTATION_UPDATE_RATE 16
// local define for the gword representation update rate
#define GWORD_REPRESENTATION_UPDATE_RATE 4
// local define for potential local word weight sum threshold
#define LWORD_WEIGHT_SUM_THRESHOLD 1.0f
// local define for potential global word weight sum threshold
#define GWORD_WEIGHT_SUM_THRESHOLD 1.0f
// local define for the gword decimation factor
#define GWORD_WEIGHT_DECIMATION_FACTOR 0.9f
// local define for the amount of weight offset to apply to words, making sure new words aren't always better than old ones
#define LWORD_WEIGHT_OFFSET 2500
// local define for the initial weight of a new word (used to make sure old words aren't worse off than new seeds)
#define LWORD_INIT_WEIGHT (1.0f/LWORD_WEIGHT_OFFSET)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorCBLBSP::BackgroundSubtractorCBLBSP(	 float fLBSPThreshold
														,size_t nInitDescDistThreshold
														,size_t nInitColorDistThreshold
														,float fLocalWordsPerChannel
														,float fGlobalWordsPerChannel)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
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
	m_oDistThresholdVariationFrame = cv::Scalar(1.0f);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(BGSCBLBSP_T_LOWER);
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
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fLBSPThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT); // @@@@@ use a*x+b instead of just a*x?
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
			for(size_t n=0; n<(s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight); ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oImgSize.width*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar nSampleColor = m_oLastColorFrame.data[idx_sample_color];
				const ushort nSampleIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_1ch* pCurrLocalWord = ((LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx]);
					if(pCurrLocalWord
							&& absdiff_uchar(nSampleColor,pCurrLocalWord->nColor)<=nColorDistThreshold
							&& ((popcount_ushort_8bitsLUT(pCurrLocalWord->nDesc)<s_nDescMaxDataRange_1ch/2)?hdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc):gdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc))<=nDescDistThreshold) {
						++pCurrLocalWord->nOccurrences;
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
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET;
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
					double fDevalFactor = (double)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords;
					pCurrLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*std::pow(fDevalFactor,2));
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(nColorDistThreshold+1))-(int)nColorDistThreshold/2;
					pCurrLocalWord->nColor = cv::saturate_cast<uchar>((int)pRefLocalWord->nColor+nRandColorOffset);
					pCurrLocalWord->nDesc = pRefLocalWord->nDesc;
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
				if(absdiff_uchar(pCurrGlobalWord->nColor,pRefBestLocalWord->nColor)<=nColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(pRefBestLocalWord->nDesc)-(int)pCurrGlobalWord->nDescBITS)<=nDescDistThreshold)
					break;
				++nGlobalWordIdx;
			}
			if(nGlobalWordIdx==nGlobalWordFillIdx) {
				pCurrGlobalWord = apGlobalWordListIter++;
				pCurrGlobalWord->nColor = pRefBestLocalWord->nColor;
				pCurrGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(pRefBestLocalWord->nDesc);
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				++nGlobalWordFillIdx;
			}
			m_apGlobalWordLookupTable[idx_orig_uchar] = pCurrGlobalWord;
			pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
			*(float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_float) += fRefBestLocalWordWeight;
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
							if(absdiff_uchar(pCurrGlobalWord->nColor,pRefLocalWord->nColor)<=nColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(pRefLocalWord->nDesc)-(int)pCurrGlobalWord->nDescBITS)<=nDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==nGlobalWordFillIdx) {
							pCurrGlobalWord = apGlobalWordListIter++;
							pCurrGlobalWord->nColor = pRefLocalWord->nColor;
							pCurrGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(pRefLocalWord->nDesc);
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
							m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							++nGlobalWordFillIdx;
						}
						m_apGlobalWordLookupTable[nLookupMapIdx] = pCurrGlobalWord;
						pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
						*(float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4)) += fRefLocalWordWeight;
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
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fLBSPThreshold); // @@@@@ use a*x+b instead of just a*x?
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
			for(size_t n=0; n<(s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight); ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = (m_oImgSize.width*y_sample + x_sample)*3;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar* const anSampleColor = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const anSampleIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nLocalWordIdx;
				for(nLocalWordIdx=0;nLocalWordIdx<m_nLocalWords;++nLocalWordIdx) {
					LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nLocalWordIdx];
					if(pCurrLocalWord
							&& L1dist_uchar(anSampleColor,pCurrLocalWord->anColor)<=nTotColorDistThreshold
							&& ((popcount_ushort_8bitsLUT(pCurrLocalWord->anDesc)<s_nDescMaxDataRange_3ch/2)?hdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc):gdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc))<=nTotDescDistThreshold) {
						++pCurrLocalWord->nOccurrences;
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
					}
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET;
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
					double fDevalFactor = (double)(m_nLocalWords-nLocalWordIdx)/m_nLocalWords;
					pCurrLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*std::pow(fDevalFactor,2));
					const size_t nRandLocalWordIdx = (rand()%nLocalWordIdx);
					const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nRandLocalWordIdx];
					const int nRandColorOffset = (rand()%(m_nColorDistThreshold+1))-(int)m_nColorDistThreshold/2;
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = cv::saturate_cast<uchar>((int)pRefLocalWord->anColor[c]+nRandColorOffset);
						pCurrLocalWord->anDesc[c] = pRefLocalWord->anDesc[c];
					}
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
				if(L1dist_uchar(pCurrGlobalWord->anColor,pRefBestLocalWord->anColor)<=nTotColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(pRefBestLocalWord->anDesc)-(int)pCurrGlobalWord->nDescBITS)<=nTotDescDistThreshold)
					break;
				++nGlobalWordIdx;
			}
			if(nGlobalWordIdx==nGlobalWordFillIdx) {
				pCurrGlobalWord = apGlobalWordListIter++;
				for(size_t c=0; c<3; ++c)
					pCurrGlobalWord->anColor[c] = pRefBestLocalWord->anColor[c];
				pCurrGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(pRefBestLocalWord->anDesc);
				pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
				pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
				pCurrGlobalWord->fLatestWeight = 0.0f;
				m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
				++nGlobalWordFillIdx;
			}
			m_apGlobalWordLookupTable[idx_orig_uchar] = pCurrGlobalWord;
			pCurrGlobalWord->fLatestWeight += fRefBestLocalWordWeight;
			*(float*)(pCurrGlobalWord->oSpatioOccMap.data+idx_orig_float) += fRefBestLocalWordWeight;
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
							if(L1dist_uchar(pCurrGlobalWord->anColor,pRefLocalWord->anColor)<=nTotColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(pRefLocalWord->anDesc)-(int)pCurrGlobalWord->nDescBITS)<=nTotDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==nGlobalWordFillIdx) {
							pCurrGlobalWord = apGlobalWordListIter++;
							for(size_t c=0; c<3; ++c)
								pCurrGlobalWord->anColor[c] = pRefLocalWord->anColor[c];
							pCurrGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(pRefLocalWord->anDesc);
							pCurrGlobalWord->oSpatioOccMap.create(m_oImgSize,CV_32FC1);
							pCurrGlobalWord->oSpatioOccMap = cv::Scalar(0.0f);
							pCurrGlobalWord->fLatestWeight = 0.0f;
							m_apGlobalDict[nGlobalWordIdx] = pCurrGlobalWord;
							++nGlobalWordFillIdx;
						}
						m_apGlobalWordLookupTable[nLookupMapIdx] = pCurrGlobalWord;
						pCurrGlobalWord->fLatestWeight += fRefLocalWordWeight;
						*(float*)(pCurrGlobalWord->oSpatioOccMap.data+(nLookupMapIdx*4)) += fRefLocalWordWeight;
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
	++m_nFrameIndex;
	const size_t nKeyPoints = m_voKeyPoints.size();
	const size_t nCurrGlobalWordUpdateRate = /*learningRateOverride>0?(size_t)ceil(learningRateOverride):*/GWORD_REPRESENTATION_UPDATE_RATE;
#if DISPLAY_CBLBSP_DEBUG_INFO
	std::vector<std::string> vsWordModList(m_nMaxLocalDictionaries*m_nLocalWords);
	uchar anDBGColor[3] = {0,0,0};
	ushort anDBGIntraDesc[3] = {0,0,0};
	bool bDBGMaskResult = false;
	size_t idx_dbg_ldict = UINT_MAX;
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
			size_t nMinDescDist=s_nDescMaxDataRange_1ch;
			size_t nMinColorDist=s_nColorMaxDataRange_1ch;
			size_t nMinSumDist=s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = m_nDescDistThreshold;//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				const uchar& nCurrLocalWordColor = pCurrLocalWord->nColor;
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nCurrLocalWordColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nCurrLocalWordIntraDesc = pCurrLocalWord->nDesc;
					size_t nIntraDescDist, nInterDescDist;
					if(popcount_ushort_8bitsLUT(nCurrLocalWordIntraDesc)<s_nDescMaxDataRange_1ch/2) {
						nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,nCurrLocalWordIntraDesc);
						if(nIntraDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
						LBSP::computeGrayscaleDescriptor(oInputImg,nCurrLocalWordColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrLocalWordColor],nCurrInterDesc);
						nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,nCurrLocalWordIntraDesc);
						if(nInterDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
					}
					else {
						nIntraDescDist = gdist_ushort_8bitLUT(nCurrIntraDesc,nCurrLocalWordIntraDesc);
						if(nIntraDescDist>nCurrDescDistThreshold)
							goto failedcheck1ch;
						LBSP::computeGrayscaleDescriptor(oInputImg,nCurrLocalWordColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrLocalWordColor],nCurrInterDesc);
						nInterDescDist = gdist_ushort_8bitLUT(nCurrInterDesc,nCurrLocalWordIntraDesc);
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
					++pCurrLocalWord->nOccurrences;
					fPotentialLocalWordsWeightSum += GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					if(!m_oFGMask_last.data[idx_uchar] && nDescDist<=nCurrDescDistThreshold/2 && nColorDist>=nCurrColorDistThreshold/2 && (rand()%nCurrLocalWordUpdateRate)==0) { // @@@@ using (intra+inter)/2...
						pCurrLocalWord->nColor = nCurrColor;
						//pCurrLocalWord->nDesc = nCurrIntraDesc; @@@@@
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
					if(nMinDescDist>nDescDist)
						nMinDescDist = nDescDist;
					if(nMinColorDist>nColorDist)
						nMinColorDist = nColorDist;
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
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(absdiff_uchar(nLastColor,nCurrColor))/s_nColorMaxDataRange_1ch+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/s_nDescMaxDataRange_1ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN; // @@@@ add bit trick?
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			if(fPotentialLocalWordsWeightSum>=LWORD_WEIGHT_SUM_THRESHOLD) {
				// == background
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1))/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
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
					const size_t n_rand = rand();
					const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate)); // @@@@ use neighbor's update rate?
					if((n_rand%nCurrLocalWordNeighborSpreadRate)==0 || (fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) { // @@@@@
						size_t nRandLocalWordIdx = 0;
						float fPotentialRandLocalWordsWeightSum = 0.0f;
						while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							if(absdiff_uchar(nCurrColor,pRandLocalWord->nColor)<=nCurrColorDistThreshold/2 // @@@ thrs/2
									&& ((popcount_ushort_8bitsLUT(pRandLocalWord->nDesc)<s_nDescMaxDataRange_1ch/2)?hdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc):gdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc))<=nCurrDescDistThreshold/2) {
								++pRandLocalWord->nOccurrences;
								fPotentialRandLocalWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
#if DISPLAY_CBLBSP_DEBUG_INFO
								vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
							}
							++nRandLocalWordIdx;
						}
						if(fPotentialRandLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							nRandLocalWordIdx = m_nLocalWords-1;
							LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							pRandLocalWord->nColor = nCurrColor;
							pRandLocalWord->nDesc = nCurrIntraDesc;
							pRandLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*(LWORD_WEIGHT_SUM_THRESHOLD-fPotentialRandLocalWordsWeightSum)/2);
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
					if(!pLastMatchedGlobalWord || absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold || (size_t)abs((int)popcount_ushort_8bitsLUT(nCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold) {
						size_t nGlobalWordIdx = 0;
						while(nGlobalWordIdx<m_nGlobalWords) {
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(nCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
							pLastMatchedGlobalWord->nColor = nCurrColor;
							pLastMatchedGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(nCurrIntraDesc);
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
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
				GlobalWord_1ch* pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalWordLookupTable[idx_uchar];
				if(!pLastMatchedGlobalWord || absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)>nCurrColorDistThreshold || (size_t)abs((int)popcount_ushort_8bitsLUT(nCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)>nCurrDescDistThreshold) {
					size_t nGlobalWordIdx = 0;
					while(nGlobalWordIdx<m_nGlobalWords) {
						pLastMatchedGlobalWord = (GlobalWord_1ch*)m_apGlobalDict[nGlobalWordIdx];
						if(absdiff_uchar(pLastMatchedGlobalWord->nColor,nCurrColor)<=nCurrColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(nCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)<=nCurrDescDistThreshold)
							break;
						++nGlobalWordIdx;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32)+fPotentialLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
					if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						const size_t nNewLocalWordIdx = m_nLocalWords-1;
						LocalWord_1ch* pNewLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
						pNewLocalWord->nColor = nCurrColor;
						pNewLocalWord->nDesc = nCurrIntraDesc;
						pNewLocalWord->nOccurrences = 1;
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
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
			if( ((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2))
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			else if((*pfCurrDistThresholdVariationFactor)>0) {
				(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<0)
					(*pfCurrDistThresholdVariationFactor) = 0;
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
#if DISPLAY_CBLBSP_DEBUG_INFO
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = nCurrColor;
					anDBGIntraDesc[c] = nCurrIntraDesc;
				}
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
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
			size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotColorDist=s_nColorMaxDataRange_3ch;
			size_t nMinTotSumDist = s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			const size_t nCurrLocalWordUpdateRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrTotColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const size_t nCurrTotDescDistThreshold = m_nDescDistThreshold*3;//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
			const size_t nCurrSCColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const size_t nCurrSCDescDistThreshold = (size_t)(m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			size_t nLocalWordIdx = 0;
			float fPotentialLocalWordsWeightSum = 0.0f;
			while(nLocalWordIdx<m_nLocalWords && fPotentialLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nLocalWordIdx];
				size_t nTotColorDist = 0;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				const uchar* const anCurrLocalWordColor = pCurrLocalWord->anColor;
				const ushort* const anCurrLocalWordIntraDesc = pCurrLocalWord->anDesc;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anCurrLocalWordColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					size_t nIntraDescDist, nInterDescDist;
					if(popcount_ushort_8bitsLUT(anCurrLocalWordIntraDesc[c])<s_nDescMaxDataRange_1ch/2) {
						nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,anCurrLocalWordColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anCurrLocalWordColor[c]],anCurrInterDesc[c]);
						nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nInterDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
					}
					else {
						nIntraDescDist = gdist_ushort_8bitLUT(anCurrIntraDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,anCurrLocalWordColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anCurrLocalWordColor[c]],anCurrInterDesc[c]);
						nInterDescDist = gdist_ushort_8bitLUT(anCurrInterDesc[c],anCurrLocalWordIntraDesc[c]);
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
				if(nTotColorDist<=nCurrTotColorDistThreshold && nTotDescDist<=nCurrTotDescDistThreshold && nTotSumDist<nCurrTotColorDistThreshold) {
#if DISPLAY_CBLBSP_DEBUG_INFO
					vsWordModList[idx_ldict+nLocalWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					++pCurrLocalWord->nOccurrences;
					fPotentialLocalWordsWeightSum += GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					if(!m_oFGMask_last.data[idx_uchar] && nTotDescDist<=nCurrTotDescDistThreshold/2 && nTotColorDist>=nCurrTotColorDistThreshold/2 && (rand()%nCurrLocalWordUpdateRate)==0) { // @@@@ using (intra+inter)/2...
						for(size_t c=0; c<3; ++c) {
							pCurrLocalWord->anColor[c] = anCurrColor[c];
							//pCurrLocalWord->anDesc[c] = anCurrIntraDesc[c]; @@@@@
						}
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nLocalWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
					if(nMinTotDescDist>nTotDescDist)
						nMinTotDescDist = nTotDescDist;
					if(nMinTotColorDist>nTotColorDist)
						nMinTotColorDist = nTotColorDist;
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
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(L1dist_uchar(anLastColor,anCurrColor))/s_nColorMaxDataRange_3ch+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/s_nDescMaxDataRange_3ch)/2)/BGSCBLBSP_N_SAMPLES_FOR_MEAN; // @@@@ add bit trick?
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			if(fPotentialLocalWordsWeightSum>=LWORD_WEIGHT_SUM_THRESHOLD) {
				// == background
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1))/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)>BGSCBLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSCBLBSP_T_DECR/(*pfCurrMeanMinDist);
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
					const size_t n_rand = rand();
					const size_t nCurrLocalWordNeighborSpreadRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate)); // @@@@ use neighbor's update rate?
					if((n_rand%nCurrLocalWordNeighborSpreadRate)==0 || (fRandMeanSegmRes>BGSCBLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSCBLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) { // @@@@@
						size_t nRandLocalWordIdx = 0;
						float fPotentialRandLocalWordsWeightSum = 0.0f;
						while(nRandLocalWordIdx<m_nLocalWords && fPotentialRandLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							if(L1dist_uchar(anCurrColor,pRandLocalWord->anColor)<=nCurrTotColorDistThreshold/2 // @@@ thrs/2
									&& ((popcount_ushort_8bitsLUT(pRandLocalWord->anDesc)<s_nDescMaxDataRange_3ch/2)?hdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc):gdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc))<=nCurrTotDescDistThreshold/2) {
								++pRandLocalWord->nOccurrences;
								fPotentialRandLocalWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
#if DISPLAY_CBLBSP_DEBUG_INFO
								vsWordModList[idx_rand_ldict+nRandLocalWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
							}
							++nRandLocalWordIdx;
						}
						if(fPotentialRandLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							nRandLocalWordIdx = m_nLocalWords-1;
							LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandLocalWordIdx];
							for(size_t c=0; c<3; ++c) {
								pRandLocalWord->anColor[c] = anCurrColor[c];
								pRandLocalWord->anDesc[c] = anCurrIntraDesc[c];
							}
							pRandLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*(LWORD_WEIGHT_SUM_THRESHOLD-fPotentialRandLocalWordsWeightSum)/2);
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
					if(!pLastMatchedGlobalWord || L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)>nCurrTotColorDistThreshold || (size_t)abs((int)popcount_ushort_8bitsLUT(anCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold) {
						size_t nGlobalWordIdx = 0;
						while(nGlobalWordIdx<m_nGlobalWords) {
							pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							if(L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)<=nCurrTotColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(anCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold)
								break;
							++nGlobalWordIdx;
						}
						if(nGlobalWordIdx==m_nGlobalWords) {
							nGlobalWordIdx = m_nGlobalWords-1;
							pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
							for(size_t c=0; c<3; ++c)
								pLastMatchedGlobalWord->anColor[c] = anCurrColor[c];
							pLastMatchedGlobalWord->nDescBITS = popcount_ushort_8bitsLUT(anCurrIntraDesc);
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
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSCBLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSCBLBSP_N_SAMPLES_FOR_MEAN;
				if((*pfCurrLearningRate)<BGSCBLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSCBLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSCBLBSP_T_UPPER)
						*pfCurrLearningRate = BGSCBLBSP_T_UPPER;
				}
				GlobalWord_3ch* pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalWordLookupTable[idx_uchar];
				if(!pLastMatchedGlobalWord || L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)>nCurrTotColorDistThreshold || (size_t)abs((int)popcount_ushort_8bitsLUT(anCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)>nCurrTotDescDistThreshold) {
					size_t nGlobalWordIdx = 0;
					while(nGlobalWordIdx<m_nGlobalWords) {
						pLastMatchedGlobalWord = (GlobalWord_3ch*)m_apGlobalDict[nGlobalWordIdx];
						if(L1dist_uchar(pLastMatchedGlobalWord->anColor,anCurrColor)<=nCurrTotColorDistThreshold && (size_t)abs((int)popcount_ushort_8bitsLUT(anCurrIntraDesc)-(int)pLastMatchedGlobalWord->nDescBITS)<=nCurrTotDescDistThreshold)
							break;
						++nGlobalWordIdx;
					}
					if(nGlobalWordIdx==m_nGlobalWords)
						pLastMatchedGlobalWord = nullptr;
					m_apGlobalWordLookupTable[idx_uchar] = pLastMatchedGlobalWord;
				}
				if(!pLastMatchedGlobalWord || *(float*)(pLastMatchedGlobalWord->oSpatioOccMap.data+idx_flt32)+fPotentialLocalWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
					oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
					if(fPotentialLocalWordsWeightSum<=LWORD_INIT_WEIGHT) {
						const size_t nNewLocalWordIdx = m_nLocalWords-1;
						LocalWord_3ch* pNewLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nNewLocalWordIdx];
						for(size_t c=0; c<3; ++c) {
							pNewLocalWord->anColor[c] = anCurrColor[c];
							pNewLocalWord->anDesc[c] = anCurrIntraDesc[c];
						}
						pNewLocalWord->nOccurrences = 1;
						pNewLocalWord->nFirstOcc = m_nFrameIndex;
						pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_INFO
						vsWordModList[idx_ldict+nNewLocalWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_INFO
					}
				}
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
			if( ((*pfCurrMeanMinDist)>BGSCBLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN) ||
				((*pfCurrMeanSegmRes)>BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2))
				(*pfCurrDistThresholdVariationFactor) += BGSCBLBSP_R2_INCR;
			else if((*pfCurrDistThresholdVariationFactor)>0) {
				(*pfCurrDistThresholdVariationFactor) -= BGSCBLBSP_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<0)
					(*pfCurrDistThresholdVariationFactor) = 0;
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
			}
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		}
	}
#if DISPLAY_CBLBSP_DEBUG_INFO
	cv::Mat gwords_coverage(m_oImgSize,CV_32FC1);
	gwords_coverage = cv::Scalar(0.0f);
	for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords; ++nDBGWordIdx)
		cv::max(gwords_coverage,m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap,gwords_coverage);
	cv::imshow("gwords_coverage",gwords_coverage);
	std::string asDBGStrings[5] = {"gword[0]","gword[1]","gword[2]","gword[3]","gword[4]"};
	for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nGlobalWords && nDBGWordIdx<5; ++nDBGWordIdx)
		cv::imshow(asDBGStrings[nDBGWordIdx],m_apGlobalDict[nDBGWordIdx]->oSpatioOccMap);
	double minVal,maxVal;
	cv::minMaxIdx(gwords_coverage,&minVal,&maxVal);
	std::cout << " " << m_nFrameIndex << " : gwords_coverage min=" << minVal << ", max=" << maxVal << std::endl;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
	for(size_t nGlobalWordIdx=1; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx)
		if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight>m_apGlobalDict[nGlobalWordIdx-1]->fLatestWeight)
			std::swap(m_apGlobalDict[nGlobalWordIdx],m_apGlobalDict[nGlobalWordIdx-1]);
	/*
	if(!(m_nFrameIndex%nCurrGlobalWordUpdateRate)) {
#if DISPLAY_CBLBSP_DEBUG_INFO
		std::cout << "\tBlurring gword occurrence maps..." << std::endl;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
	*/
		for(size_t nGlobalWordIdx=0; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx) {
			if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight==0.0f)
				continue;
			//cv::imshow("gword_oSpatioOccMap",m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap);
			//cv::GaussianBlur(m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,cv::Size(7,7),0,0,cv::BORDER_REPLICATE);
			cv::blur(m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap,cv::Size(7,7),cv::Point(-1,-1),cv::BORDER_REPLICATE);
			//cv::imshow("gword_oSpatioOccMap_blurred",m_apGlobalDict[nGlobalWordIdx]->oSpatioOccMap);
			//cv::waitKey(0);
		}
	//}
	if(!(m_nFrameIndex%(nCurrGlobalWordUpdateRate))) {
#if DISPLAY_CBLBSP_DEBUG_INFO
		std::cout << "\tDecimating gword weights..." << std::endl;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		// INCORPORATE DECIMATE TO THE MAIN KEYPOINT LOOP??? @@@@@@@@@@@@@@@@@@@@
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
			}
		}
	}
	if(!(m_nFrameIndex%2048)) {
#if DISPLAY_CBLBSP_DEBUG_INFO
		std::cout << "\tRecalculating gword weights to correct drift..." << std::endl;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
		for(size_t nGlobalWordIdx=0; nGlobalWordIdx<m_nGlobalWords; ++nGlobalWordIdx) {
			if(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight==0.0f)
				continue;
#if DISPLAY_CBLBSP_DEBUG_INFO
			float fDBGWordWeight = GetGlobalWordWeight(m_apGlobalDict[nGlobalWordIdx]);
			std::cout << "\t\tgword[" << nGlobalWordIdx << "] -- calc=" << m_apGlobalDict[nGlobalWordIdx]->fLatestWeight << ", true=" << fDBGWordWeight << " (" << (std::abs(m_apGlobalDict[nGlobalWordIdx]->fLatestWeight-fDBGWordWeight)/m_apGlobalDict[nGlobalWordIdx]->fLatestWeight)*100.0f << "% diff)" << std::endl;
#endif //DISPLAY_CBLBSP_DEBUG_INFO
			m_apGlobalDict[nGlobalWordIdx]->fLatestWeight = GetGlobalWordWeight(m_apGlobalDict[nGlobalWordIdx]);
		}
	}
#if DISPLAY_CBLBSP_DEBUG_INFO
	if(idx_dbg_ldict!=UINT_MAX) {
		std::cout << std::endl;
		cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
		printf("\nDBG[%2d,%2d] : \n",nDebugCoordX,nDebugCoordY);
		printf("\t Color=[%03d,%03d,%03d]\n",(int)anDBGColor[0],(int)anDBGColor[1],(int)anDBGColor[2]);
		printf("\t IntraDesc=[%05d,%05d,%05d], IntraDesc_BITS=[%02lu,%02lu,%02lu]\n",anDBGIntraDesc[0],anDBGIntraDesc[1],anDBGIntraDesc[2],popcount_ushort_8bitsLUT(anDBGIntraDesc[0]),popcount_ushort_8bitsLUT(anDBGIntraDesc[1]),popcount_ushort_8bitsLUT(anDBGIntraDesc[2]));
		printf("\t FG_Mask=[%s]\n",(bDBGMaskResult?"TRUE":"FALSE"));
		printf("----\n");
		printf("DBG_LDICT : \n");
		for(size_t nDBGWordIdx=0; nDBGWordIdx<m_nLocalWords; ++nDBGWordIdx) {
			if(m_nImgChannels==1) {
				LocalWord_1ch* pDBGLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_dbg_ldict+nDBGWordIdx];
				printf("\t [%02lu] : weight=[%02.03f], nColor=[%03d], nDescBITS=[%02lu]  %s\n",nDBGWordIdx,GetLocalWordWeight(pDBGLocalWord,m_nFrameIndex),(int)pDBGLocalWord->nColor,popcount_ushort_8bitsLUT(pDBGLocalWord->nDesc),vsWordModList[idx_dbg_ldict+nDBGWordIdx].c_str());
			}
			else { //m_nImgChannels==3
				LocalWord_3ch* pDBGLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_dbg_ldict+nDBGWordIdx];
				printf("\t [%02lu] : weight=[%02.03f], anColor=[%03d,%03d,%03d], anDescBITS=[%02lu,%02lu,%02lu]  %s\n",nDBGWordIdx,GetLocalWordWeight(pDBGLocalWord,m_nFrameIndex),(int)pDBGLocalWord->anColor[0],(int)pDBGLocalWord->anColor[1],(int)pDBGLocalWord->anColor[2],popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[0]),popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[1]),popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[2]),vsWordModList[idx_dbg_ldict+nDBGWordIdx].c_str());
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
