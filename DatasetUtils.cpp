#include "DatasetUtils.h"

CategoryInfo::CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname)
	:	 m_sName(name)
		,m_sDBName(dbname) {
	if(m_sDBName==CDNET_DB_NAME || m_sDBName==WALLFLOWER_DB_NAME /*|| m_sDBName==...*/) {
		// all subdirs are considered sequence directories for this category; no parsing is done for files at this level.
		std::vector<std::string> vsSequencePaths;
		GetSubDirsFromDir(dir,vsSequencePaths);
		for(size_t i=0; i<vsSequencePaths.size(); i++) {
			size_t pos = vsSequencePaths[i].find_last_of("/\\");
			if(pos==std::string::npos)
				m_vpSequences.push_back(new SequenceInfo(vsSequencePaths[i],vsSequencePaths[i],dbname,this));
			else
				m_vpSequences.push_back(new SequenceInfo(vsSequencePaths[i].substr(pos+1),vsSequencePaths[i],dbname,this));
		}
	}
	else
		throw std::runtime_error(std::string("Unknown database name, cannot use any known parsing strategy."));
}

CategoryInfo::~CategoryInfo() {
	for(size_t i=0; i<m_vpSequences.size(); i++)
		delete m_vpSequences[i];
}

SequenceInfo::SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent)
	:	 m_sName(name)
		,m_sDBName(dbname)
		,m_pParent(parent)
		,m_nIMReadInputFlags((parent->m_sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR) // force thermal sequences to be loaded as grayscale (faster processing, better noise compensation)){
		,m_nTestGTIndex(UINT_MAX) { // only used for sequences with only one GT frame (e.g. WALLFLOWER)
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
		m_oROI = cv::imread(dir+"/ROI.bmp");
		if(m_oROI.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess a ROI.bmp file.");
		m_oSize = m_oROI.size();
	}
	else if(m_sDBName==WALLFLOWER_DB_NAME) {
		std::vector<std::string> vsImgPaths;
		GetFilesFromDir(dir,vsImgPaths);
		bool bFoundScript=false, bFoundGTFile=false;
		for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
			if(*iter==dir+"/script.txt")
				bFoundScript = true;
			else if(iter->find("hand_segmented_")!=std::string::npos) {
				m_nTestGTIndex = atoi(iter->substr(iter->find("hand_segmented_")+15,5).c_str());
				bFoundGTFile = true;
				m_vsGTFramePaths.push_back(*iter);
			}
			else {
				if(iter->find(".bmp")!=iter->size()-4)
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
	}
	/*else if(m_sDBName==...) {
		// ...
	}*/
	else
		throw std::runtime_error(std::string("Unknown database name, cannot use any known parsing strategy."));
}

size_t SequenceInfo::GetNbInputFrames() const {
	return m_vsInputFramePaths.size();
}

size_t SequenceInfo::GetNbGTFrames() const {
	return m_vsGTFramePaths.size();
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
	else /*if(m_sDBName==...) {
		// ...
	}*/
		return cv::Mat();
}

cv::Mat SequenceInfo::GetGTFrameFromIndex(size_t idx) {
	if(m_sDBName==CDNET_DB_NAME) {
		CV_DbgAssert(idx>=0 && idx<m_vsInputFramePaths.size());
		return cv::imread(m_vsGTFramePaths[idx],cv::IMREAD_GRAYSCALE);
	}
	else if(m_sDBName==WALLFLOWER_DB_NAME) {
		CV_DbgAssert(m_vsGTFramePaths.size()==1);
		if(idx==m_nTestGTIndex)
			return cv::imread(m_vsGTFramePaths[idx],cv::IMREAD_GRAYSCALE);
		else
			return cv::Mat(m_oSize,CV_8UC1,cv::Scalar(128));
	}
	else /*if(m_sDBName==...) {
		// ...
	}*/
		return cv::Mat();
}
