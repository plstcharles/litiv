#include "DatasetUtils.h"

CategoryInfo::CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname)
	:	 m_sName(name)
		,m_sDBName(dbname)
		,nTP(0)
		,nTN(0)
		,nFP(0)
		,nFN(0) {
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
			m_vpSequences.push_back(new SequenceInfo(*iter,*iter,dbname,this));
		else
			m_vpSequences.push_back(new SequenceInfo(iter->substr(pos+1),*iter,dbname,this));
	}
}

CategoryInfo::~CategoryInfo() {
	for(size_t i=0; i<m_vpSequences.size(); i++)
		delete m_vpSequences[i];
}

SequenceInfo::SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent)
	:	 m_sName(name)
		,m_sDBName(dbname)
		,nTP(0)
		,nTN(0)
		,nFP(0)
		,nFN(0)
		,m_pParent(parent)
		,m_nIMReadInputFlags((parent->m_sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR) { // force thermal sequences to be loaded as grayscale (faster processing, better noise compensation))
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
		m_oROI = cv::Mat::ones(oTempImg.size(),oTempImg.type());
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
		m_oROI = cv::Mat::ones(oTempImg.size(),oTempImg.type());
		m_oSize = oTempImg.size();
		m_nNextFrame = 0;
		m_nTotalNbFrames = (size_t)m_voVideoReader.get(CV_CAP_PROP_FRAME_COUNT);
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name, cannot use any known parsing strategy."));
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

cv::Mat SequenceInfo::GetInputFrameFromIndex(size_t idx) {
	if(m_sDBName==CDNET_DB_NAME) {
		CV_DbgAssert(idx>=0 && idx<m_vsInputFramePaths.size());
		return cv::imread(m_vsInputFramePaths[idx],m_nIMReadInputFlags);
	}
	else if(m_sDBName==WALLFLOWER_DB_NAME) {
		CV_DbgAssert(idx>=0 && idx<m_vsInputFramePaths.size());
		return cv::imread(m_vsInputFramePaths[idx],m_nIMReadInputFlags);
	}
	else if(m_sDBName==PETS2001_D3TC1_DB_NAME) {
		CV_DbgAssert(idx>=0 && idx<m_voVideoReader.get(CV_CAP_PROP_FRAME_COUNT));
		cv::Mat oFrame;
		if(m_nNextFrame!=idx)
			m_voVideoReader.set(CV_CAP_PROP_POS_FRAMES,idx);
		m_voVideoReader >> oFrame;
		++m_nNextFrame;
		return oFrame;
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name."));
}

cv::Mat SequenceInfo::GetGTFrameFromIndex(size_t idx) {
	if(m_sDBName==CDNET_DB_NAME) {
		CV_DbgAssert(idx>=0 && idx<m_vsInputFramePaths.size());
		return cv::imread(m_vsGTFramePaths[idx],cv::IMREAD_GRAYSCALE);
	}
	else if(m_sDBName==WALLFLOWER_DB_NAME || m_sDBName==PETS2001_D3TC1_DB_NAME) {
		auto res = m_mTestGTIndexes.find(idx);
		if(res!=m_mTestGTIndexes.end())
			return cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
		else
			return cv::Mat(m_oSize,CV_8UC1,cv::Scalar(128));
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name."));
}
