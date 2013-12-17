#include "BackgroundSubtractorCBLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// @@@@@@ FOR FEEDBACK LOOPS, VARYING WEIGHT THRESHOLD > VARYING DIST THRESHOLD

// local define used for debug purposes only
#define DISPLAY_CBLBSP_DEBUG_FRAMES 1
// local define for the gradient proportion value used in color+grad distance calculations
//#define OVERLOAD_GRAD_PROP ((1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSCBLBSP_R_LOWER)/(BGSCBLBSP_R_UPPER-BGSCBLBSP_R_LOWER),2))*0.5f)
// local define for the lword representation update rate
#define LWORD_REPRESENTATION_UPDATE_RATE 16
// local define for potential word weight sum threshold
#define LWORD_WEIGHT_SUM_THRESHOLD 1.0f
// local define for the replaceable lword fraction
#define LWORD_REPLACEABLE_FRAC 8
// local define for the amount of weight offset to apply to words, making sure new words aren't always better than old ones
#define LWORD_WEIGHT_OFFSET 1024
// local define for the initial weight of a new word (used to make sure old words aren't worse off than new seeds)
#define LWORD_INIT_WEIGHT (1.0f/LWORD_WEIGHT_OFFSET)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorCBLBSP::BackgroundSubtractorCBLBSP(	 float fLBSPThreshold
														,size_t nInitDescDistThreshold
														,size_t nInitColorDistThreshold
														,size_t nLocalWords
														,size_t nGlobalWords)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_nLocalWords(nLocalWords)
		,m_nLastLocalWordReplaceableIdxs(m_nLocalWords<LWORD_REPLACEABLE_FRAC?1:(m_nLocalWords/LWORD_REPLACEABLE_FRAC))
		,m_nGlobalWords(nGlobalWords)
		,m_nLocalDictionaries(0)
		,m_aapLocalDicts(nullptr)
		,m_apLocalWordList_1ch(nullptr)
		,m_apLocalWordList_3ch(nullptr)
		,m_apGlobalDict(nullptr)
		,m_apGlobalWordList_1ch(nullptr)
		,m_apGlobalWordList_3ch(nullptr) {
	CV_Assert(m_nLocalWords>0 && m_nGlobalWords>0);
	CV_Assert(m_nLastLocalWordReplaceableIdxs>0);
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
	m_voKeyPoints = voNewKeyPoints;
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nLocalDictionaries = oInitImg.cols*oInitImg.rows;
	m_nFrameIndex = 0;
	CleanupDictionaries();
	m_aapLocalDicts = new LocalWord*[m_nLocalDictionaries*m_nLocalWords];
	memset(m_aapLocalDicts,0,sizeof(LocalWord*)*m_nLocalDictionaries*m_nLocalWords);
	m_apGlobalDict = new GlobalWord*[m_nGlobalWords];
	memset(m_apGlobalDict,0,sizeof(GlobalWord*)*m_nGlobalWords);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
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
		LocalWord_1ch* apWordListIter = m_apLocalWordList_1ch;
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			for(size_t n=0; n<(s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight*2); ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oImgSize.width*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar nSampleColor = m_oLastColorFrame.data[idx_sample_color];
				const ushort nSampleIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nWordIdx;
				for(nWordIdx=0;nWordIdx<m_nLocalWords;++nWordIdx) {
					LocalWord_1ch* pCurrLocalWord = ((LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx]);
					if(pCurrLocalWord
							&& absdiff_uchar(nSampleColor,pCurrLocalWord->nColor)<=nColorDistThreshold
							&& ((popcount_ushort_8bitsLUT(pCurrLocalWord->nDesc)<s_nDescMaxDataRange_1ch/2)?hdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc):gdist_ushort_8bitLUT(nSampleIntraDesc,pCurrLocalWord->nDesc))<=nDescDistThreshold) {
						++pCurrLocalWord->nOccurrences;
						break;
					}
				}
				if(nWordIdx==m_nLocalWords) {
					nWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
					LocalWord_1ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nWordIdx])
						pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx];
					else {
						pCurrLocalWord = apWordListIter++;
						m_aapLocalDicts[idx_orig_ldict+nWordIdx] = pCurrLocalWord;
					}
					pCurrLocalWord->nColor = nSampleColor;
					pCurrLocalWord->nDesc = nSampleIntraDesc;
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET;
				}
				while(nWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nWordIdx-1] || m_aapLocalDicts[idx_orig_ldict+nWordIdx]->nOccurrences>m_aapLocalDicts[idx_orig_ldict+nWordIdx-1]->nOccurrences)) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nWordIdx],m_aapLocalDicts[idx_orig_ldict+nWordIdx-1]);
					--nWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nWordIdx=1; nWordIdx<m_nLocalWords; ++nWordIdx) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = apWordListIter++;
					double fDevalFactor = (double)(m_nLocalWords-nWordIdx)/m_nLocalWords;
					pCurrLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*std::pow(fDevalFactor,2));
					const size_t nRandWordIdx = (rand()%nWordIdx);
					const LocalWord_1ch* pRefLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_orig_ldict+nRandWordIdx];
					const int nRandColorOffset = (rand()%(m_nColorDistThreshold+1))-(int)m_nColorDistThreshold/2;
					pCurrLocalWord->nColor = cv::saturate_cast<uchar>((int)pRefLocalWord->nColor+nRandColorOffset);
					pCurrLocalWord->nDesc = pRefLocalWord->nDesc;
					m_aapLocalDicts[idx_orig_ldict+nWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_1ch==(apWordListIter-nKeyPoints*m_nLocalWords));
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
		LocalWord_3ch* apWordListIter = m_apLocalWordList_3ch;
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			const size_t idx_orig_uchar = m_oImgSize.width*y_orig + x_orig;
			const size_t idx_orig_ldict = idx_orig_uchar*m_nLocalWords;
			for(size_t n=0; n<(s_nSamplesInitPatternWidth*s_nSamplesInitPatternHeight*2); ++n) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = (m_oImgSize.width*y_sample + x_sample)*3;
				const size_t idx_sample_desc = idx_sample_color*2;
				const uchar* const anSampleColor = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const anSampleIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
				size_t nWordIdx;
				for(nWordIdx=0;nWordIdx<m_nLocalWords;++nWordIdx) {
					LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx];
					if(pCurrLocalWord
							&& L1dist_uchar(anSampleColor,pCurrLocalWord->anColor)<=nTotColorDistThreshold
							&& ((popcount_ushort_8bitsLUT(pCurrLocalWord->anDesc)<s_nDescMaxDataRange_3ch/2)?hdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc):gdist_ushort_8bitLUT(anSampleIntraDesc,pCurrLocalWord->anDesc))<=nTotDescDistThreshold) {
						++pCurrLocalWord->nOccurrences;
						break;
					}
				}
				if(nWordIdx==m_nLocalWords) {
					nWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
					LocalWord_3ch* pCurrLocalWord;
					if(m_aapLocalDicts[idx_orig_ldict+nWordIdx])
						pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx];
					else {
						pCurrLocalWord = apWordListIter++;
						m_aapLocalDicts[idx_orig_ldict+nWordIdx] = pCurrLocalWord;
					}
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = anSampleColor[c];
						pCurrLocalWord->anDesc[c] = anSampleIntraDesc[c];
					}
					pCurrLocalWord->nOccurrences = LWORD_WEIGHT_OFFSET;
				}
				while(nWordIdx>0 && (!m_aapLocalDicts[idx_orig_ldict+nWordIdx-1] || m_aapLocalDicts[idx_orig_ldict+nWordIdx]->nOccurrences>m_aapLocalDicts[idx_orig_ldict+nWordIdx-1]->nOccurrences)) {
					std::swap(m_aapLocalDicts[idx_orig_ldict+nWordIdx],m_aapLocalDicts[idx_orig_ldict+nWordIdx-1]);
					--nWordIdx;
				}
			}
			CV_Assert(m_aapLocalDicts[idx_orig_ldict]);
			for(size_t nWordIdx=1; nWordIdx<m_nLocalWords; ++nWordIdx) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nWordIdx];
				if(!pCurrLocalWord) {
					pCurrLocalWord = apWordListIter++;
					double fDevalFactor = (double)(m_nLocalWords-nWordIdx)/m_nLocalWords;
					pCurrLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*std::pow(fDevalFactor,2));
					const size_t nRandWordIdx = (rand()%nWordIdx);
					const LocalWord_3ch* pRefLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_orig_ldict+nRandWordIdx];
					const int nRandColorOffset = (rand()%(m_nColorDistThreshold+1))-(int)m_nColorDistThreshold/2;
					for(size_t c=0; c<3; ++c) {
						pCurrLocalWord->anColor[c] = cv::saturate_cast<uchar>((int)pRefLocalWord->anColor[c]+nRandColorOffset);
						pCurrLocalWord->anDesc[c] = pRefLocalWord->anDesc[c];
					}
					m_aapLocalDicts[idx_orig_ldict+nWordIdx] = pCurrLocalWord;
				}
			}
		}
		CV_Assert(m_apLocalWordList_3ch==(apWordListIter-nKeyPoints*m_nLocalWords));
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
#if DISPLAY_CBLBSP_DEBUG_FRAMES
	std::vector<std::string> vsWordModList(m_nLocalDictionaries*m_nLocalWords);
	uchar anDBGColor[3];
	ushort anDBGIntraDesc[3];
	bool bDBGMaskResult;
	size_t idx_dbg_ldict = UINT_MAX;
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ldict = idx_uchar*m_nLocalWords;
			//const size_t idx_ushrt = idx_uchar*2;
			//const size_t idx_flt32 = idx_uchar*4;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			//size_t nMinDescDist=s_nDescMaxDataRange_1ch;
			//size_t nMinSumDist=s_nColorMaxDataRange_1ch;
			//float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			//float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			//float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			const size_t nCurrLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):LWORD_REPRESENTATION_UPDATE_RATE;//(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);//(size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = m_nDescDistThreshold;//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			size_t nWordIdx = 0;
			float fPotentialWordsWeightSum = 0.0f;
			while(nWordIdx<m_nLocalWords && fPotentialWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
				LocalWord_1ch* pCurrLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nWordIdx];
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
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					vsWordModList[idx_ldict+nWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					++pCurrLocalWord->nOccurrences;
					float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					if(fCurrLocalWordWeight>LWORD_INIT_WEIGHT)
						fPotentialWordsWeightSum += fCurrLocalWordWeight;
					else {
						pCurrLocalWord->nOccurrences = 1;
						pCurrLocalWord->nFirstOcc = m_nFrameIndex;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
					}
					if(!m_oFGMask_last.data[idx_uchar] && nIntraDescDist<=nCurrDescDistThreshold/2 && nColorDist>=nCurrColorDistThreshold/2 && (rand()%nCurrLearningRate)==0) {
						pCurrLocalWord->nColor = nCurrColor;
						//pCurrLocalWord->nDesc = nCurrIntraDesc; @@@@@
#if DISPLAY_CBLBSP_DEBUG_FRAMES
						vsWordModList[idx_ldict+nWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
					}
				}
				failedcheck1ch:
				if(nWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nWordIdx],m_aapLocalDicts[idx_ldict+nWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					std::swap(vsWordModList[idx_ldict+nWordIdx],vsWordModList[idx_ldict+nWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
				++nWordIdx;
			}
			//ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			//uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			if(fPotentialWordsWeightSum>=LWORD_WEIGHT_SUM_THRESHOLD) {
				// == background
				if((rand()%nCurrLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
					const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
					if(m_aapLocalDicts[idx_rand_ldict]) { // @@@@@ && !m_oFGMask_last.data[idx_rand_uchar]
						size_t nRandWordIdx = 0;
						float fPotentialRandWordsWeightSum = 0.0f;
						while(nRandWordIdx<m_nLocalWords && fPotentialRandWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandWordIdx];
							if(absdiff_uchar(nCurrColor,pRandLocalWord->nColor)<=nCurrColorDistThreshold/2
									&& ((popcount_ushort_8bitsLUT(pRandLocalWord->nDesc)<s_nDescMaxDataRange_1ch/2)?hdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc):gdist_ushort_8bitLUT(nCurrIntraDesc,pRandLocalWord->nDesc))<=nCurrDescDistThreshold/2) {
								++pRandLocalWord->nOccurrences;
								fPotentialRandWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
								vsWordModList[idx_rand_ldict+nRandWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
							}
							++nRandWordIdx;
						}
						if(fPotentialRandWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							nRandWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
							LocalWord_1ch* pRandLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_rand_ldict+nRandWordIdx];
							pRandLocalWord->nColor = nCurrColor;
							pRandLocalWord->nDesc = nCurrIntraDesc;
							pRandLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*(LWORD_WEIGHT_SUM_THRESHOLD-fPotentialRandWordsWeightSum)/2);
							pRandLocalWord->nFirstOcc = m_nFrameIndex;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_FRAMES
							vsWordModList[idx_rand_ldict+nRandWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
						}
					}
				}
			}
			else {
				// == foreground
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(fPotentialWordsWeightSum==0.0f) {
					const size_t nNewWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
					LocalWord_1ch* pNewLocalWord = (LocalWord_1ch*)m_aapLocalDicts[idx_ldict+nNewWordIdx];
					pNewLocalWord->nColor = nCurrColor;
					pNewLocalWord->nDesc = nCurrIntraDesc;
					pNewLocalWord->nOccurrences = 1;
					pNewLocalWord->nFirstOcc = m_nFrameIndex;
					pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					vsWordModList[idx_ldict+nNewWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
			}
			while(nWordIdx<m_nLocalWords) {
				if(nWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nWordIdx],m_aapLocalDicts[idx_ldict+nWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					std::swap(vsWordModList[idx_ldict+nWordIdx],vsWordModList[idx_ldict+nWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
				++nWordIdx;
			}
			//nLastIntraDesc = nCurrIntraDesc;
			//nLastColor = nCurrColor;
#if DISPLAY_CBLBSP_DEBUG_FRAMES
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = nCurrColor;
					anDBGIntraDesc[c] = nCurrIntraDesc;
				}
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
			}
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ldict = idx_uchar*m_nLocalWords;
			//const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			//const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			//size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			//size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
			//float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			//float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			//float* pfCurrWeightThreshold = ((float*)(m_oWeightThresholdFrame.data+idx_flt32));
			//float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			const size_t nCurrLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):LWORD_REPRESENTATION_UPDATE_RATE;//(size_t)ceil((*pfCurrLearningRate));
			const size_t nCurrTotColorDistThreshold = m_nColorDistThreshold*3;//(size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const size_t nCurrTotDescDistThreshold = m_nDescDistThreshold*3;//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const size_t nCurrSCColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);//(size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const size_t nCurrSCDescDistThreshold = (size_t)(m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);//(size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			size_t nWordIdx = 0;
			float fPotentialWordsWeightSum = 0.0f;
			while(nWordIdx<m_nLocalWords && fPotentialWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
				LocalWord_3ch* pCurrLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nWordIdx];
				size_t nTotColorDist = 0;
				size_t nTotIntraDescDist = 0;
				size_t nTotInterDescDist = 0;
				const uchar* const anCurrLocalWordColor = pCurrLocalWord->anColor;
				const ushort* const anCurrLocalWordIntraDesc = pCurrLocalWord->anDesc;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anCurrLocalWordColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					if(popcount_ushort_8bitsLUT(anCurrLocalWordIntraDesc[c])<s_nDescMaxDataRange_1ch/2) {
						const size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nIntraDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,anCurrLocalWordColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anCurrLocalWordColor[c]],anCurrInterDesc[c]);
						const size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nInterDescDist>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						nTotIntraDescDist += nIntraDescDist;
						nTotInterDescDist += nInterDescDist;
					}
					else {
						const size_t nIntraDescDist_BITS = gdist_ushort_8bitLUT(anCurrIntraDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nIntraDescDist_BITS>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						LBSP::computeSingleRGBDescriptor(oInputImg,anCurrLocalWordColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anCurrLocalWordColor[c]],anCurrInterDesc[c]);
						const size_t nInterDescDist_BITS = gdist_ushort_8bitLUT(anCurrInterDesc[c],anCurrLocalWordIntraDesc[c]);
						if(nInterDescDist_BITS>nCurrSCDescDistThreshold)
							goto failedcheck3ch;
						nTotIntraDescDist += nIntraDescDist_BITS;
						nTotInterDescDist += nInterDescDist_BITS;
					}
					nTotColorDist += nColorDist;
				}
				if(nTotInterDescDist<=nCurrTotDescDistThreshold && nTotIntraDescDist<=nCurrTotDescDistThreshold && nTotColorDist<=nCurrTotColorDistThreshold) {
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					vsWordModList[idx_ldict+nWordIdx] += "MATCHED ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
					pCurrLocalWord->nLastOcc = m_nFrameIndex;
					++pCurrLocalWord->nOccurrences;
					float fCurrLocalWordWeight = GetLocalWordWeight(pCurrLocalWord,m_nFrameIndex);
					if(fCurrLocalWordWeight>LWORD_INIT_WEIGHT)
						fPotentialWordsWeightSum += fCurrLocalWordWeight;
					else {
						pCurrLocalWord->nOccurrences = 1;
						pCurrLocalWord->nFirstOcc = m_nFrameIndex;
						pCurrLocalWord->nLastOcc = m_nFrameIndex;
					}
					if(!m_oFGMask_last.data[idx_uchar] && nTotIntraDescDist<=nCurrTotDescDistThreshold/2 && nTotColorDist>=nCurrTotColorDistThreshold/2 && (rand()%nCurrLearningRate)==0) {
						for(size_t c=0; c<3; ++c) {
							pCurrLocalWord->anColor[c] = anCurrColor[c];
							//pCurrLocalWord->anDesc[c] = anCurrIntraDesc[c]; @@@@@
						}
#if DISPLAY_CBLBSP_DEBUG_FRAMES
						vsWordModList[idx_ldict+nWordIdx] += "UPDATED ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
					}
				}
				failedcheck3ch:
				if(nWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nWordIdx],m_aapLocalDicts[idx_ldict+nWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					std::swap(vsWordModList[idx_ldict+nWordIdx],vsWordModList[idx_ldict+nWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
				++nWordIdx;
			}
			//ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			//uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			if(fPotentialWordsWeightSum>=LWORD_WEIGHT_SUM_THRESHOLD) {
				// == background
				if((rand()%nCurrLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t idx_rand_uchar = (m_oImgSize.width*y_rand + x_rand);
					const size_t idx_rand_ldict = idx_rand_uchar*m_nLocalWords;
					if(m_aapLocalDicts[idx_rand_ldict]) { // @@@@@ && !m_oFGMask_last.data[idx_rand_uchar]
						size_t nRandWordIdx = 0;
						float fPotentialRandWordsWeightSum = 0.0f;
						while(nRandWordIdx<m_nLocalWords && fPotentialRandWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandWordIdx];
							if(L1dist_uchar(anCurrColor,pRandLocalWord->anColor)<=nCurrTotColorDistThreshold/2
									&& ((popcount_ushort_8bitsLUT(pRandLocalWord->anDesc)<s_nDescMaxDataRange_3ch/2)?hdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc):gdist_ushort_8bitLUT(anCurrIntraDesc,pRandLocalWord->anDesc))<=nCurrTotDescDistThreshold/2) {
								++pRandLocalWord->nOccurrences;
								fPotentialRandWordsWeightSum += GetLocalWordWeight(pRandLocalWord,m_nFrameIndex);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
								vsWordModList[idx_rand_ldict+nRandWordIdx] += "MATCHED(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
							}
							++nRandWordIdx;
						}
						if(fPotentialRandWordsWeightSum<LWORD_WEIGHT_SUM_THRESHOLD) {
							nRandWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
							LocalWord_3ch* pRandLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_rand_ldict+nRandWordIdx];
							for(size_t c=0; c<3; ++c) {
								pRandLocalWord->anColor[c] = anCurrColor[c];
								pRandLocalWord->anDesc[c] = anCurrIntraDesc[c];
							}
							pRandLocalWord->nOccurrences = (size_t)(LWORD_WEIGHT_OFFSET*(LWORD_WEIGHT_SUM_THRESHOLD-fPotentialRandWordsWeightSum)/2);
							pRandLocalWord->nFirstOcc = m_nFrameIndex;
							pRandLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_FRAMES
							vsWordModList[idx_rand_ldict+nRandWordIdx] += "NEW(NEIGHBOR) ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
						}
					}
				}
			}
			else {
				// == foreground
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(fPotentialWordsWeightSum==0.0f) {
					const size_t nNewWordIdx = m_nLocalWords-(rand()%m_nLastLocalWordReplaceableIdxs)-1;
					LocalWord_3ch* pNewLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_ldict+nNewWordIdx];
					for(size_t c=0; c<3; ++c) {
						pNewLocalWord->anColor[c] = anCurrColor[c];
						pNewLocalWord->anDesc[c] = anCurrIntraDesc[c];
					}
					pNewLocalWord->nOccurrences = 1;
					pNewLocalWord->nFirstOcc = m_nFrameIndex;
					pNewLocalWord->nLastOcc = m_nFrameIndex;
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					vsWordModList[idx_ldict+nNewWordIdx] += "NEW ";
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
			}
			while(nWordIdx<m_nLocalWords) {
				if(nWordIdx>0 && GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx],m_nFrameIndex)>GetLocalWordWeight(m_aapLocalDicts[idx_ldict+nWordIdx-1],m_nFrameIndex)) {
					std::swap(m_aapLocalDicts[idx_ldict+nWordIdx],m_aapLocalDicts[idx_ldict+nWordIdx-1]);
#if DISPLAY_CBLBSP_DEBUG_FRAMES
					std::swap(vsWordModList[idx_ldict+nWordIdx],vsWordModList[idx_ldict+nWordIdx-1]);
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
				}
				++nWordIdx;
			}
			//for(size_t c=0; c<3; ++c) {
			//	anLastIntraDesc[c] = anCurrIntraDesc[c];
			//	anLastColor[c] = anCurrColor[c];
			//}
#if DISPLAY_CBLBSP_DEBUG_FRAMES
			if(y==nDebugCoordY && x==nDebugCoordX) {
				for(size_t c=0; c<3; ++c) {
					anDBGColor[c] = anCurrColor[c];
					anDBGIntraDesc[c] = anCurrIntraDesc[c];
				}
				bDBGMaskResult = (oCurrFGMask.data[idx_uchar]==UCHAR_MAX);
				idx_dbg_ldict = idx_ldict;
			}
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
		}
	}
#if DISPLAY_CBLBSP_DEBUG_FRAMES
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
			LocalWord_3ch* pDBGLocalWord = (LocalWord_3ch*)m_aapLocalDicts[idx_dbg_ldict+nDBGWordIdx];
			printf("\t [%02lu] : weight=[%02.03f], anColor=[%03d,%03d,%03d], anDescBITS=[%02lu,%02lu,%02lu]  %s\n",nDBGWordIdx,GetLocalWordWeight(pDBGLocalWord,m_nFrameIndex),(int)pDBGLocalWord->anColor[0],(int)pDBGLocalWord->anColor[1],(int)pDBGLocalWord->anColor[2],popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[0]),popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[1]),popcount_ushort_8bitsLUT(pDBGLocalWord->anDesc[2]),vsWordModList[idx_dbg_ldict+nDBGWordIdx].c_str());
		}
		/*cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
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
		cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_R_UPPER,-BGSPBASLBSP_R_LOWER/BGSPBASLBSP_R_UPPER);
		cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,cv::Size(320,240));
		cv::imshow("r(x)",oDistThresholdFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oDistThresholdVariationFrameNormalized; cv::normalize(m_oDistThresholdVariationFrame,oDistThresholdVariationFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
		cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(255));
		cv::resize(oDistThresholdVariationFrameNormalized,oDistThresholdVariationFrameNormalized,cv::Size(320,240));
		cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << "r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
		cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_T_UPPER,-BGSPBASLBSP_T_LOWER/BGSPBASLBSP_T_UPPER);
		cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
		cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,cv::Size(320,240));
		cv::imshow("t(x)",oUpdateRateFrameNormalized);
		std::cout << std::fixed << std::setprecision(5) << " t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;*/
	}
#endif //DISPLAY_CBLBSP_DEBUG_FRAMES
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
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
	if(m_apGlobalWordList_1ch) {
		delete[] m_apGlobalWordList_1ch;
		m_apGlobalWordList_1ch = nullptr;
	}
	if(m_apGlobalWordList_3ch) {
		delete[] m_apGlobalWordList_3ch;
		m_apGlobalWordList_3ch = nullptr;
	}
	if(m_aapLocalDicts) {
		delete[] m_aapLocalDicts;
		m_aapLocalDicts = nullptr;
	}
	if(m_apGlobalDict) {
		delete[] m_apGlobalDict;
		m_apGlobalDict = nullptr;
	}
}

float BackgroundSubtractorCBLBSP::GetLocalWordWeight(const LocalWord* w, size_t nCurrFrame) {
	return (float)(w->nOccurrences)/((w->nLastOcc-w->nFirstOcc)/2+(nCurrFrame-w->nLastOcc)/4+LWORD_WEIGHT_OFFSET);
}

float BackgroundSubtractorCBLBSP::GetGlobalWordWeight(const GlobalWord* /*w*/, size_t /*nCurrFrame*/) {
	return -1; //@@@@
}

BackgroundSubtractorCBLBSP::LocalWord::~LocalWord() {}
BackgroundSubtractorCBLBSP::GlobalWord::~GlobalWord() {}
