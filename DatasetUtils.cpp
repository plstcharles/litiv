#include "DatasetUtils.h"

CategoryInfo::CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname, bool forceGrayscale)
	:	 m_sName(name)
		,m_sDBName(dbname)
		,nTP(0)
		,nTN(0)
		,nFP(0)
		,nFN(0)
		,nSE(0)
		,m_dAvgFPS(-1) {
	std::cout << "\tParsing dir '" << dir << "' for category '" << name << "'... ";
	std::vector<std::string> vsSequencePaths;
	if(m_sDBName==CDNET_DB_NAME || m_sDBName==WALLFLOWER_DB_NAME || m_sDBName==PETS2001_D3TC1_DB_NAME/*|| m_sDBName==...*/) {
		// all subdirs are considered sequence directories
		GetSubDirsFromDir(dir,vsSequencePaths);
		std::cout << "(" << vsSequencePaths.size() << " subdir sequences)" << std::endl;
	}
	/*else if(m_sDBName==...) {
			// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name, cannot use any known parsing strategy."));
	for(auto iter=vsSequencePaths.begin(); iter!=vsSequencePaths.end(); ++iter) {
		size_t pos = iter->find_last_of("/\\");
		if(pos==std::string::npos)
			m_vpSequences.push_back(new SequenceInfo(*iter,*iter,dbname,this,forceGrayscale));
		else
			m_vpSequences.push_back(new SequenceInfo(iter->substr(pos+1),*iter,dbname,this,forceGrayscale));
	}
}

CategoryInfo::~CategoryInfo() {
	for(size_t i=0; i<m_vpSequences.size(); i++)
		delete m_vpSequences[i];
}

SequenceInfo::SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent, bool forceGrayscale)
	:	 m_sName(name)
		,m_sDBName(dbname)
		,nTP(0)
		,nTN(0)
		,nFP(0)
		,nFN(0)
		,nSE(0)
		,m_dAvgFPS(-1)
#if USE_PRECACHED_IO
		,m_bIsPrecaching(false)
		,m_nNextExpectedInputFrameIdx(0)
		,m_nNextExpectedGTFrameIdx(0)
		,m_nNextPrecachedInputFrameIdx(0)
		,m_nNextPrecachedGTFrameIdx(0)
#else //!USE_PRECACHED_IO
		,m_nLastReqInputFrameIndex(UINT_MAX)
		,m_nLastReqGTFrameIndex(UINT_MAX)
#endif //!USE_PRECACHED_IO
		,m_nNextExpectedVideoReaderFrameIdx(0)
		,m_nTotalNbFrames(0)
		,m_pParent(parent)
		,m_nIMReadInputFlags(forceGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR) {
	if(m_sDBName==CDNET_DB_NAME) {
		std::vector<std::string> vsSubDirs;
		GetSubDirsFromDir(dir,vsSubDirs);
		auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),dir+"/groundtruth");
		auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),dir+"/input");
		if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess the required groundtruth and input directories.");
		GetFilesFromDir(*inputDir,m_vsInputFramePaths);
		GetFilesFromDir(*gtDir,m_vsGTFramePaths);
		if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess same amount of GT & input frames.");
		m_oROI = cv::imread(dir+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
		if(m_oROI.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess a ROI.bmp file.");
		m_oSize = m_oROI.size();
		m_nTotalNbFrames = m_vsInputFramePaths.size();
		// note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
	}
	else if(m_sDBName==WALLFLOWER_DB_NAME) {
		std::vector<std::string> vsImgPaths;
		GetFilesFromDir(dir,vsImgPaths);
		bool bFoundScript=false, bFoundGTFile=false;
		const std::string sGTFilePrefix("hand_segmented_");
		const size_t nInputFileNbDecimals = 5;
		const std::string sInputFileSuffix(".bmp");
		for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
			if(*iter==dir+"/script.txt")
				bFoundScript = true;
			else if(iter->find(sGTFilePrefix)!=std::string::npos) {
				m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),m_vsGTFramePaths.size()));
				m_vsGTFramePaths.push_back(*iter);
				bFoundGTFile = true;
			}
			else {
				if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
					throw std::runtime_error(std::string("Sequence directory at ") + dir + " contained an unknown file ('" + *iter + "')");
				m_vsInputFramePaths.push_back(*iter);
			}
		}
		if(!bFoundGTFile || !bFoundScript || m_vsInputFramePaths.empty() || m_vsGTFramePaths.size()!=1)
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess the required groundtruth and input files.");
		cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
		if(oTempImg.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess a valid GT file.");
		m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(VAL_POSITIVE));
		m_oSize = oTempImg.size();
		m_nTotalNbFrames = m_vsInputFramePaths.size();
	}
	else if(m_sDBName==PETS2001_D3TC1_DB_NAME) {
		std::vector<std::string> vsVideoSeqPaths;
		GetFilesFromDir(dir,vsVideoSeqPaths);
		if(vsVideoSeqPaths.size()!=1)
			throw std::runtime_error(std::string("Bad subdirectory ('")+dir+std::string("') for PETS2001 parsing (should contain only one video sequence file)"));
		std::vector<std::string> vsGTSubdirPaths;
		GetSubDirsFromDir(dir,vsGTSubdirPaths);
		if(vsGTSubdirPaths.size()!=1)
			throw std::runtime_error(std::string("Bad subdirectory ('")+dir+std::string("') for PETS2001 parsing (should contain only one GT subdir)"));
		m_voVideoReader.open(vsVideoSeqPaths[0]);
		if(!m_voVideoReader.isOpened())
			throw std::runtime_error(std::string("Bad video file ('")+vsVideoSeqPaths[0]+std::string("'), could not be opened."));
		GetFilesFromDir(vsGTSubdirPaths[0],m_vsGTFramePaths);
		if(m_vsGTFramePaths.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess any valid GT frames.");
		const std::string sGTFilePrefix("image_");
		const size_t nInputFileNbDecimals = 4;
		for(auto iter=m_vsGTFramePaths.begin(); iter!=m_vsGTFramePaths.end(); ++iter)
			m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),iter-m_vsGTFramePaths.begin()));
		cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
		if(oTempImg.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess valid GT file(s).");
		m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(VAL_POSITIVE));
		m_oSize = oTempImg.size();
		m_nNextExpectedVideoReaderFrameIdx = 0;
		m_nTotalNbFrames = (size_t)m_voVideoReader.get(CV_CAP_PROP_FRAME_COUNT);
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name, cannot use any known parsing strategy."));
#if USE_PRECACHED_IO
	CV_Assert(MAX_NB_PRECACHED_FRAMES>1);
	CV_Assert(PRECACHE_REFILL_THRESHOLD>1 && PRECACHE_REFILL_THRESHOLD<MAX_NB_PRECACHED_FRAMES);
	CV_Assert(REQUEST_TIMEOUT_MS>0);
	CV_Assert(QUERY_TIMEOUT_MS>0);
#endif //USE_PRECACHED_IO
}

SequenceInfo::~SequenceInfo() {
#if USE_PRECACHED_IO
	StopPrecaching();
#endif //USE_PRECACHED_IO
}

size_t SequenceInfo::GetNbInputFrames() const {
	return m_nTotalNbFrames;
}

size_t SequenceInfo::GetNbGTFrames() const {
	return m_mTestGTIndexes.size();
}

cv::Size SequenceInfo::GetFrameSize() const {
	return m_oSize;
}

cv::Mat SequenceInfo::GetSequenceROI() const {
	return m_oROI;
}

void SequenceInfo::ValidateKeyPoints(std::vector<cv::KeyPoint>& voKPs) const {
	std::vector<cv::KeyPoint> voNewKPs;
	for(size_t k=0; k<voKPs.size(); ++k) {
		if(m_oROI.at<uchar>(voKPs[k].pt)>0)
			voNewKPs.push_back(voKPs[k]);
	}
	voKPs = voNewKPs;
}

cv::Mat SequenceInfo::GetInputFrameFromIndex_Internal(size_t idx) {
	CV_DbgAssert(idx<m_nTotalNbFrames);
	cv::Mat oFrame;
	if(m_sDBName==CDNET_DB_NAME || m_sDBName==WALLFLOWER_DB_NAME)
		oFrame = cv::imread(m_vsInputFramePaths[idx],m_nIMReadInputFlags);
	else if(m_sDBName==PETS2001_D3TC1_DB_NAME) {
		if(m_nNextExpectedVideoReaderFrameIdx!=idx)
			m_voVideoReader.set(CV_CAP_PROP_POS_FRAMES,idx);
		m_voVideoReader >> oFrame;
		++m_nNextExpectedVideoReaderFrameIdx;
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	CV_DbgAssert(oFrame.size()==m_oSize);
	return oFrame;
}

cv::Mat SequenceInfo::GetGTFrameFromIndex_Internal(size_t idx) {
	CV_DbgAssert(idx<m_nTotalNbFrames);
	cv::Mat oFrame;
	if(m_sDBName==CDNET_DB_NAME)
		oFrame = cv::imread(m_vsGTFramePaths[idx],cv::IMREAD_GRAYSCALE);
	else if(m_sDBName==WALLFLOWER_DB_NAME || m_sDBName==PETS2001_D3TC1_DB_NAME) {
		auto res = m_mTestGTIndexes.find(idx);
		if(res!=m_mTestGTIndexes.end())
			oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
		else
			oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar(VAL_OUTOFSCOPE));
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	CV_DbgAssert(oFrame.size()==m_oSize);
	return oFrame;
}

const cv::Mat& SequenceInfo::GetInputFrameFromIndex(size_t idx) {
#if USE_PRECACHED_IO
	if(!m_bIsPrecaching)
		throw std::runtime_error(m_sName + " [SequenceInfo] : Error, queried a frame before precaching was activated.");
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> sync_lock(m_oInputFrameSyncMutex);
	m_nRequestInputFrameIndex = idx;
	std::cv_status res;
	do {
		m_oInputFrameReqCondVar.notify_one();
		res = m_oInputFrameSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(REQUEST_TIMEOUT_MS));
		//if(res==std::cv_status::timeout) std::cout << " # retrying request..." << std::endl;
	} while(res==std::cv_status::timeout);
	return m_oReqInputFrame;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
	EnterCriticalSection(&m_oInputFrameSyncMutex);
	m_nRequestInputFrameIndex = idx;
	BOOL res;
	do {
		WakeConditionVariable(&m_oInputFrameReqCondVar);
		res = SleepConditionVariableCS(&m_oInputFrameSyncCondVar,&m_oInputFrameSyncMutex,REQUEST_TIMEOUT_MS);
		//if(!res) std::cout << " # retrying request..." << std::endl;
	} while(!res);
	LeaveCriticalSection(&m_oInputFrameSyncMutex);
	return m_oReqInputFrame;
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#else //!USE_PRECACHED_IO
	if(m_nLastReqInputFrameIndex!=idx) {
		m_oLastReqInputFrame = GetInputFrameFromIndex_Internal(idx);
		m_nLastReqInputFrameIndex = idx;
	}
	return m_oLastReqInputFrame;
#endif //!USE_PRECACHED_IO
}

const cv::Mat& SequenceInfo::GetGTFrameFromIndex(size_t idx) {
#if USE_PRECACHED_IO
	if(!m_bIsPrecaching)
		throw std::runtime_error(m_sName + " [SequenceInfo] : Error, queried a frame before precaching was activated.");
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> sync_lock(m_oGTFrameSyncMutex);
	m_nRequestGTFrameIndex = idx;
	std::cv_status res;
	do {
		m_oGTFrameReqCondVar.notify_one();
		res = m_oGTFrameSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(REQUEST_TIMEOUT_MS));
		//if(res==std::cv_status::timeout) std::cout << " # retrying request..." << std::endl;
	} while(res==std::cv_status::timeout);
	return m_oReqGTFrame;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
	EnterCriticalSection(&m_oGTFrameSyncMutex);
	m_nRequestGTFrameIndex = idx;
	BOOL res;
	do {
		WakeConditionVariable(&m_oGTFrameReqCondVar);
		res = SleepConditionVariableCS(&m_oGTFrameSyncCondVar,&m_oGTFrameSyncMutex,REQUEST_TIMEOUT_MS);
		//if(!res) std::cout << " # retrying request..." << std::endl;
	} while(!res);
	LeaveCriticalSection(&m_oGTFrameSyncMutex);
	return m_oReqGTFrame;
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#else //!USE_PRECACHED_IO
	if(m_nLastReqGTFrameIndex!=idx) {
		m_oLastReqGTFrame = GetGTFrameFromIndex_Internal(idx);
		m_nLastReqGTFrameIndex = idx;
	}
	return m_oLastReqGTFrame;
#endif //!USE_PRECACHED_IO
}

#if USE_PRECACHED_IO

void SequenceInfo::PrecacheInputFrames() {
	srand((size_t)time(NULL)*m_nTotalNbFrames*m_sName.size());
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> sync_lock(m_oInputFrameSyncMutex);
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
	EnterCriticalSection(&m_oInputFrameSyncMutex);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
	size_t nInitFramesToPrecache = MAX_NB_PRECACHED_FRAMES/2 + rand()%(MAX_NB_PRECACHED_FRAMES/2);
	//std::cout << " @ initializing precaching with " << nInitFramesToPrecache << " frames " << std::endl;
	while(m_qoInputFrameCache.size()<nInitFramesToPrecache && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames)
		m_qoInputFrameCache.push_back(GetInputFrameFromIndex_Internal(m_nNextPrecachedInputFrameIdx++));
	while(m_bIsPrecaching) {
#if PLATFORM_SUPPORTS_CPP11
		if(m_oInputFrameReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
		if(SleepConditionVariableCS(&m_oInputFrameReqCondVar,&m_oInputFrameSyncMutex,QUERY_TIMEOUT_MS)) {
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
			CV_DbgAssert(m_nRequestInputFrameIndex<m_nTotalNbFrames);
			if(m_nRequestInputFrameIndex!=m_nNextExpectedInputFrameIdx-1) {
				if(!m_qoInputFrameCache.empty() && m_nRequestInputFrameIndex==m_nNextExpectedInputFrameIdx) {
					m_oReqInputFrame = m_qoInputFrameCache.front();
					m_qoInputFrameCache.pop_front();
				}
				else {
					if(!m_qoInputFrameCache.empty()) {
						//std::cout << " @ answering request manually, out of order (req=" << m_nRequestInputFrameIndex << ", expected=" << m_nNextExpectedInputFrameIdx <<") ";
						CV_DbgAssert((m_nNextPrecachedInputFrameIdx-m_qoInputFrameCache.size())==m_nNextExpectedInputFrameIdx);
						if(m_nRequestInputFrameIndex<m_nNextPrecachedInputFrameIdx && m_nRequestInputFrameIndex>m_nNextExpectedInputFrameIdx) {
							//std::cout << " -- popping " << m_nRequestInputFrameIndex-m_nNextExpectedInputFrameIdx << " item(s) from cache" << std::endl;
							while(m_nRequestInputFrameIndex-m_nNextExpectedInputFrameIdx>0) {
								m_qoInputFrameCache.pop_front();
								++m_nNextExpectedInputFrameIdx;
							}
							m_oReqInputFrame = m_qoInputFrameCache.front();
							m_qoInputFrameCache.pop_front();
						}
						else {
							//std::cout << " -- destroying cache" << std::endl;
							m_qoInputFrameCache.clear();
							m_oReqInputFrame = GetInputFrameFromIndex_Internal(m_nRequestInputFrameIndex);
							m_nNextPrecachedInputFrameIdx = m_nRequestInputFrameIndex+1;
						}
					}
					else {
						//std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
						m_oReqInputFrame = GetInputFrameFromIndex_Internal(m_nRequestInputFrameIndex);
						m_nNextPrecachedInputFrameIdx = m_nRequestInputFrameIndex+1;
					}
				}
			}
			//else std::cout << " @ answering request using last frame" << std::endl;
			m_nNextExpectedInputFrameIdx = m_nRequestInputFrameIndex+1;
#if PLATFORM_SUPPORTS_CPP11
			m_oInputFrameSyncCondVar.notify_one();
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
			WakeConditionVariable(&m_oInputFrameSyncCondVar);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
		}
		else {
			CV_DbgAssert((m_nNextPrecachedInputFrameIdx-m_nNextExpectedInputFrameIdx)==m_qoInputFrameCache.size());
			if(m_qoInputFrameCache.size()<PRECACHE_REFILL_THRESHOLD && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames) {
				//std::cout << " @ filling precache buffer... (" << MAX_NB_PRECACHED_FRAMES-m_qoInputFrameCache.size() << " frames)" << std::endl;
				while(m_qoInputFrameCache.size()<MAX_NB_PRECACHED_FRAMES && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames)
					m_qoInputFrameCache.push_back(GetInputFrameFromIndex_Internal(m_nNextPrecachedInputFrameIdx++));
			}
		}
	}
#if !PLATFORM_SUPPORTS_CPP11 && PLATFORM_USES_WIN32API
	LeaveCriticalSection(&m_oInputFrameSyncMutex);
#endif //!PLATFORM_SUPPORTS_CPP11 && PLATFORM_USES_WIN32API
}

void SequenceInfo::PrecacheGTFrames() {
	srand((size_t)time(NULL)*m_nTotalNbFrames*m_sName.size());
#if PLATFORM_SUPPORTS_CPP11
	std::unique_lock<std::mutex> sync_lock(m_oGTFrameSyncMutex);
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
	EnterCriticalSection(&m_oGTFrameSyncMutex);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
	size_t nInitFramesToPrecache = PRECACHE_REFILL_THRESHOLD/2 + rand()%(MAX_NB_PRECACHED_FRAMES/2);
	//std::cout << " @ initializing precaching with " << nInitFramesToPrecache << " frames " << std::endl;
	while(m_qoGTFrameCache.size()<nInitFramesToPrecache && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames)
		m_qoGTFrameCache.push_back(GetGTFrameFromIndex_Internal(m_nNextPrecachedGTFrameIdx++));
	while(m_bIsPrecaching) {
#if PLATFORM_SUPPORTS_CPP11
		if(m_oGTFrameReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
		if(SleepConditionVariableCS(&m_oGTFrameReqCondVar,&m_oGTFrameSyncMutex,QUERY_TIMEOUT_MS)) {
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
			CV_DbgAssert(m_nRequestGTFrameIndex<m_nTotalNbFrames);
			if(m_nRequestGTFrameIndex!=m_nNextExpectedGTFrameIdx-1) {
				if(!m_qoGTFrameCache.empty() && m_nRequestGTFrameIndex==m_nNextExpectedGTFrameIdx) {
					m_oReqGTFrame = m_qoGTFrameCache.front();
					m_qoGTFrameCache.pop_front();
				}
				else {
					if(!m_qoGTFrameCache.empty()) {
						//std::cout << " @ answering request manually, out of order (req=" << m_nRequestGTFrameIndex << ", expected=" << m_nNextExpectedGTFrameIdx <<") ";
						CV_DbgAssert((m_nNextPrecachedGTFrameIdx-m_qoGTFrameCache.size())==m_nNextExpectedGTFrameIdx);
						if(m_nRequestGTFrameIndex<m_nNextPrecachedGTFrameIdx && m_nRequestGTFrameIndex>m_nNextExpectedGTFrameIdx) {
							//std::cout << " -- popping " << m_nRequestGTFrameIndex-m_nNextExpectedGTFrameIdx << " item(s) from cache" << std::endl;
							while(m_nRequestGTFrameIndex-m_nNextExpectedGTFrameIdx>0) {
								m_qoGTFrameCache.pop_front();
								++m_nNextExpectedGTFrameIdx;
							}
							m_oReqGTFrame = m_qoGTFrameCache.front();
							m_qoGTFrameCache.pop_front();
						}
						else {
							//std::cout << " -- destroying cache" << std::endl;
							m_qoGTFrameCache.clear();
							m_oReqGTFrame = GetGTFrameFromIndex_Internal(m_nRequestGTFrameIndex);
							m_nNextPrecachedGTFrameIdx = m_nRequestGTFrameIndex+1;
						}
					}
					else {
						//std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
						m_oReqGTFrame = GetGTFrameFromIndex_Internal(m_nRequestGTFrameIndex);
						m_nNextPrecachedGTFrameIdx = m_nRequestGTFrameIndex+1;
					}
				}
			}
			//else std::cout << " @ answering request using last frame" << std::endl;
			m_nNextExpectedGTFrameIdx = m_nRequestGTFrameIndex+1;
#if PLATFORM_SUPPORTS_CPP11
			m_oGTFrameSyncCondVar.notify_one();
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
			WakeConditionVariable(&m_oGTFrameSyncCondVar);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
		}
		else {
			CV_DbgAssert((m_nNextPrecachedGTFrameIdx-m_nNextExpectedGTFrameIdx)==m_qoGTFrameCache.size());
			if(m_qoGTFrameCache.size()<PRECACHE_REFILL_THRESHOLD && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames) {
				//std::cout << " @ filling precache buffer... (" << MAX_NB_PRECACHED_FRAMES-m_qoGTFrameCache.size() << " frames)" << std::endl;
				while(m_qoGTFrameCache.size()<MAX_NB_PRECACHED_FRAMES && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames)
					m_qoGTFrameCache.push_back(GetGTFrameFromIndex_Internal(m_nNextPrecachedGTFrameIdx++));
			}
		}
	}
#if !PLATFORM_SUPPORTS_CPP11 && PLATFORM_USES_WIN32API
	LeaveCriticalSection(&m_oGTFrameSyncMutex);
#endif //!PLATFORM_SUPPORTS_CPP11 && PLATFORM_USES_WIN32API
}

void SequenceInfo::StartPrecaching() {
	if(!m_bIsPrecaching) {
		m_bIsPrecaching = true;
#if PLATFORM_SUPPORTS_CPP11
		m_hInputFramePrecacher = std::thread(&SequenceInfo::PrecacheInputFrames,this);
		m_hGTFramePrecacher = std::thread(&SequenceInfo::PrecacheGTFrames,this);
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
		InitializeCriticalSection(&m_oInputFrameSyncMutex);
		InitializeCriticalSection(&m_oGTFrameSyncMutex);
		InitializeConditionVariable(&m_oInputFrameReqCondVar);
		InitializeConditionVariable(&m_oGTFrameReqCondVar);
		InitializeConditionVariable(&m_oInputFrameSyncCondVar);
		InitializeConditionVariable(&m_oGTFrameSyncCondVar);
		m_hInputFramePrecacher = CreateThread(NULL,NULL,&SequenceInfo::PrecacheInputFramesEntryPoint,(LPVOID)this,0,NULL);
		m_hGTFramePrecacher = CreateThread(NULL,NULL,&SequenceInfo::PrecacheGTFramesEntryPoint,(LPVOID)this,0,NULL);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
	}
}

void SequenceInfo::StopPrecaching() {
	if(m_bIsPrecaching) {
		m_bIsPrecaching = false;
#if PLATFORM_SUPPORTS_CPP11
		m_hInputFramePrecacher.join();
		m_hGTFramePrecacher.join();
#elif PLATFORM_USES_WIN32API //!PLATFORM_SUPPORTS_CPP11
		//CloseHandle();
		WaitForSingleObject(m_hInputFramePrecacher,INFINITE);
		WaitForSingleObject(m_hGTFramePrecacher,INFINITE);
		CloseHandle(m_hInputFramePrecacher);
		CloseHandle(m_hGTFramePrecacher);
		DeleteCriticalSection(&m_oInputFrameSyncMutex);
		DeleteCriticalSection(&m_oGTFrameSyncMutex);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for precached io support on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
	}
}

#endif //USE_PRECACHED_IO

AdvancedMetrics::AdvancedMetrics(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN, uint64_t /*nSE*/)
	:	 dRecall((double)nTP/(nTP+nFN))
		,dSpecficity((double)nTN/(nTN+nFP))
		,dFPR((double)nFP/(nFP+nTN))
#if USE_BROKEN_FNR_FUNCTION
		,dFNR((double)nFN/(nTN+nFP))
#else //!USE_BROKEN_FNR_FUNCTION
		,dFNR((double)nFN/(nTP+nFN))
#endif //!USE_BROKEN_FNR_FUNCTION
		,dPBC(100.0*(nFN+nFP)/(nTP+nFP+nFN+nTN))
		,dPrecision((double)nTP/(nTP+nFP))
		,dFMeasure(2.0*(dRecall*dPrecision)/(dRecall+dPrecision))
		,bAveraged(false) {}

AdvancedMetrics::AdvancedMetrics(const SequenceInfo* pSeq)
	:	 dRecall((double)pSeq->nTP/(pSeq->nTP+pSeq->nFN))
		,dSpecficity((double)pSeq->nTN/(pSeq->nTN+pSeq->nFP))
		,dFPR((double)pSeq->nFP/(pSeq->nFP+pSeq->nTN))
#if USE_BROKEN_FNR_FUNCTION
		,dFNR((double)pSeq->nFN/(pSeq->nTN+pSeq->nFP))
#else //!USE_BROKEN_FNR_FUNCTION
		,dFNR((double)pSeq->nFN/(pSeq->nTP+pSeq->nFN))
#endif //!USE_BROKEN_FNR_FUNCTION
		,dPBC(100.0*(pSeq->nFN+pSeq->nFP)/(pSeq->nTP+pSeq->nFP+pSeq->nFN+pSeq->nTN))
		,dPrecision((double)pSeq->nTP/(pSeq->nTP+pSeq->nFP))
		,dFMeasure(2.0*(dRecall*dPrecision)/(dRecall+dPrecision))
		,dFPS(pSeq->m_dAvgFPS)
		,bAveraged(false) {}

AdvancedMetrics::AdvancedMetrics(const CategoryInfo* pCat, bool bAverage)
	:	 bAveraged(bAverage) {
	CV_Assert(!pCat->m_vpSequences.empty());
	if(!bAverage) {
		dRecall = ((double)pCat->nTP/(pCat->nTP+pCat->nFN));
		dSpecficity = ((double)pCat->nTN/(pCat->nTN+pCat->nFP));
		dFPR = ((double)pCat->nFP/(pCat->nFP+pCat->nTN));
#if USE_BROKEN_FNR_FUNCTION
		dFNR = ((double)pCat->nFN/(pCat->nTN+pCat->nFP));
#else //!USE_BROKEN_FNR_FUNCTION
		dFNR = ((double)pCat->nFN/(pCat->nTP+pCat->nFN));
#endif //!USE_BROKEN_FNR_FUNCTION
		dPBC = (100.0*(pCat->nFN+pCat->nFP)/(pCat->nTP+pCat->nFP+pCat->nFN+pCat->nTN));
		dPrecision = ((double)pCat->nTP/(pCat->nTP+pCat->nFP));
		dFMeasure = (2.0*(dRecall*dPrecision)/(dRecall+dPrecision));
		dFPS = pCat->m_dAvgFPS;
	}
	else {
		dRecall = 0;
		dSpecficity = 0;
		dFPR = 0;
		dFNR = 0;
		dPBC = 0;
		dPrecision = 0;
		dFMeasure = 0;
		dFPS = 0;
		const size_t nSeq = pCat->m_vpSequences.size();
		for(size_t i=0; i<nSeq; ++i) {
			AdvancedMetrics temp(pCat->m_vpSequences[i]);
			dRecall += temp.dRecall;
			dSpecficity += temp.dSpecficity;
			dFPR += temp.dFPR;
			dFNR += temp.dFNR;
			dPBC += temp.dPBC;
			dPrecision += temp.dPrecision;
			dFMeasure += temp.dFMeasure;
			dFPS += temp.dFPS;
		}
		dRecall /= nSeq;
		dSpecficity /= nSeq;
		dFPR /= nSeq;
		dFNR /= nSeq;
		dPBC /= nSeq;
		dPrecision /= nSeq;
		dFMeasure /= nSeq;
		dFPS /= nSeq;
	}
}

AdvancedMetrics::AdvancedMetrics(const std::vector<CategoryInfo*>& vpCat, bool bAverage)
	:	 bAveraged(bAverage) {
	CV_Assert(!vpCat.empty());
	const size_t nCat = vpCat.size();
	if(!bAverage) {
		uint64_t nGlobalTP=0, nGlobalTN=0, nGlobalFP=0, nGlobalFN=0, nGlobalSE=0;
		dFPS=0;
		for(size_t i=0; i<nCat; ++i) {
			nGlobalTP += vpCat[i]->nTP;
			nGlobalTN += vpCat[i]->nTN;
			nGlobalFP += vpCat[i]->nFP;
			nGlobalFN += vpCat[i]->nFN;
			nGlobalSE += vpCat[i]->nSE;
			dFPS += vpCat[i]->m_dAvgFPS;
		}
		dRecall = ((double)nGlobalTP/(nGlobalTP+nGlobalFN));
		dSpecficity = ((double)nGlobalTN/(nGlobalTN+nGlobalFP));
		dFPR = ((double)nGlobalFP/(nGlobalFP+nGlobalTN));
#if USE_BROKEN_FNR_FUNCTION
		dFNR = ((double)nGlobalFN/(nGlobalTN+nGlobalFP));
#else //!USE_BROKEN_FNR_FUNCTION
		dFNR = ((double)nGlobalFN/(nGlobalTP+nGlobalFN));
#endif //!USE_BROKEN_FNR_FUNCTION
		dPBC = (100.0*(nGlobalFN+nGlobalFP)/(nGlobalTP+nGlobalFP+nGlobalFN+nGlobalTN));
		dPrecision = ((double)nGlobalTP/(nGlobalTP+nGlobalFP));
		dFMeasure = (2.0*(dRecall*dPrecision)/(dRecall+dPrecision));
		dFPS /= nCat;
	}
	else {
		dRecall = 0;
		dSpecficity = 0;
		dFPR = 0;
		dFNR = 0;
		dPBC = 0;
		dPrecision = 0;
		dFMeasure = 0;
		dFPS = 0;
		for(size_t i=0; i<nCat; ++i) {
			AdvancedMetrics temp(vpCat[i],true);
			dRecall += temp.dRecall;
			dSpecficity += temp.dSpecficity;
			dFPR += temp.dFPR;
			dFNR += temp.dFNR;
			dPBC += temp.dPBC;
			dPrecision += temp.dPrecision;
			dFMeasure += temp.dFMeasure;
			dFPS += temp.dFPS;
		}
		dRecall /= nCat;
		dSpecficity /= nCat;
		dFPR /= nCat;
		dFNR /= nCat;
		dPBC /= nCat;
		dPrecision /= nCat;
		dFMeasure /= nCat;
		dFPS /= nCat;
	}
}
