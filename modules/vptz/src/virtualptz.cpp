// (see header for licensing information)
#include "litiv/vptz/virtualptz.hpp"

inline std::string GetRootFolderPath(const std::string& sPath) {
    const size_t nLastFrwdSlashPos = sPath.find_last_of('/');
    const size_t nLastBackSlashPos = sPath.find_last_of('\\');
    const size_t nMinLastSeparatorPos = std::min(nLastFrwdSlashPos,nLastBackSlashPos);
    const size_t nMaxLastSeparatorPos = std::max(nLastFrwdSlashPos,nLastBackSlashPos);
    const size_t nLastSeparatorPos = nMaxLastSeparatorPos==sPath.npos?nMinLastSeparatorPos:nMaxLastSeparatorPos;
    return nLastSeparatorPos==sPath.npos?std::string():(sPath.substr(0,nLastSeparatorPos)+"/");
}

vptz::Camera::Camera( const std::string& sInputPath, double verti_FOV, double output_width,
                      double output_height, double hori_angle, double verti_angle,
                      double hori_speed, double verti_speed, double communication_delay) :
        m_sInputPath(sInputPath) {
    lvDbgExceptionWatch;
    panoImage = cv::imread(m_sInputPath);
    isVideo = panoImage.empty();
    if(isVideo) {
        lvAssert__(panoCapture.open(m_sInputPath),"cannot open the input panomaric video file at %s",m_sInputPath.c_str());
        m_nScenarioFrameCount = (int)panoCapture.get(cv::CAP_PROP_FRAME_COUNT);
        m_dFrameRate = panoCapture.get(cv::CAP_PROP_FPS);
        if(m_dFrameRate<=0 || lv::isnan(m_dFrameRate)) {
            std::cerr << "VirtualPTZ Warning : could not determine frame rate for input video; check container support on platform" << std::endl;
            m_dFrameRate = 16.0;
        }
        if(m_nScenarioFrameCount==0) {
            // note: the heavy checks here are to make sure this opencv version won't crash when seeking frames later
            if(!panoCapture.read(panoImage) || panoImage.empty())
                lvError("could not fetch first frame from image sequence");
            if(!panoCapture.read(panoImage)) // assume there is at least two frames
                lvError("could not fetch frames past the first one from image sequence");
            else if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,0) || !panoCapture.read(panoImage) || panoImage.empty())
                lvError("could not seek back inside image sequence");
            const std::string sRootFolderPath = GetRootFolderPath(m_sInputPath);
            cv::FileStorage oMetadataFS(sRootFolderPath+"metadata.yml",cv::FileStorage::READ);
            if(!oMetadataFS.isOpened())
                lvError("could not find metadata file in the image sequence folder");
            oMetadataFS["nFrameCount"] >> m_nScenarioFrameCount;
            oMetadataFS.release();
            if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,m_nScenarioFrameCount-1) || !panoCapture.read(panoImage) || panoImage.empty())
                lvError("could not seek to the last frame of the image sequence");
            if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,0) || !panoCapture.read(panoImage) || panoImage.empty())
                lvError("could not seek back to the first frame of the image sequence");
            panoCapture.set(cv::CAP_PROP_POS_FRAMES,0);
        }
        cv::Mat oTestFrame1, oTestFrame2;
        if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,0) || !panoCapture.read(oTestFrame1) || oTestFrame1.empty() || !panoCapture.read(oTestFrame2) || oTestFrame2.empty())
            lvError("could not fetch the first two frames of the image sequence");
        if(cv::countNonZero(oTestFrame1!=oTestFrame2)==0)
            lvError("opencv impl cannot properly parse the image sequence, all frames are identical");
        if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,0) || !panoCapture.read(oTestFrame2) || oTestFrame2.empty())
            lvError("could not fetch the first frame of the image sequence");
        if(cv::countNonZero(oTestFrame1!=oTestFrame2))
            lvError("opencv impl could not seek back to the first frame of the image sequence");
        panoCapture >> panoImage;
        panoCapture.set(cv::CAP_PROP_POS_FRAMES,0);
        m_nCurrFrameIdx = 0;
    }
    else {
        m_nScenarioFrameCount = 1;
        m_dFrameRate = DBL_MAX; // Keep in the first frame (frame 0);
        m_nCurrFrameIdx = 0;
    }
    if(panoImage.empty() || panoImage.type()!=CV_8UC3)
        lvError("Fetched image (or image sequence) had wrong type");
    cv::cvtColor(panoImage,panoImage,cv::COLOR_BGR2BGRA);

    // initialize the parameters
    outputWidth = std::max((int)output_width,1);
    outputHeight = std::max((int)output_height,1);
    Set(PTZ_CAM_VERTI_FOV, verti_FOV);
    Set(PTZ_CAM_HORI_ANGLE, hori_angle);
    Set(PTZ_CAM_VERTI_ANGLE, verti_angle);
    Set(PTZ_CAM_HORI_SPEED, hori_speed);
    Set(PTZ_CAM_VERTI_SPEED, verti_speed);
    Set(PTZ_CAM_COMMUNICATION_DELAY, communication_delay);
    executionDelayRatio = 1.0;
    firstTick = 0.0;
    horiAngleChange = 0.0;
    vertiAngleChange = 0.0;
    currentTime = 0.0;
    getFrameOverhead = 0.0;
    executionBeginTime = 0.0;
    executionEndTime = 0.0;
    executionDelay = 0.0;
    motionDelay = 0.0;

    // constants
    sphereGridSize = 512;

    m_pContext = std::unique_ptr<lv::gl::Context>(new lv::gl::Context(cv::Size((int)outputWidth,(int)outputHeight),"VPTZ Mapper"));

    // set the projection transformation
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(vertiFOV, outputWidth/outputHeight, 0.01, 10.0);
    glErrorCheck;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glGenTextures(1, &m_nTexID);
    glBindTexture(GL_TEXTURE_2D, m_nTexID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,panoImage.cols,panoImage.rows,0,GL_BGRA,GL_UNSIGNED_INT_8_8_8_8_REV,panoImage.data);
    glErrorCheck;
    m_pSphereObj = gluNewQuadric();
    gluQuadricDrawStyle(m_pSphereObj, GLU_FILL);           // use polygon primitives
    gluQuadricOrientation(m_pSphereObj, GLU_OUTSIDE);      // draw the normals pointing outward
    gluQuadricTexture(m_pSphereObj, GL_TRUE);              // generate texture coordinates
    m_oViewportFrame = cv::Mat(int(outputHeight), int(outputWidth), CV_8UC4);
    glErrorCheck;
}

vptz::Camera::~Camera() {
    lvDbgExceptionWatch;
    gluDeleteQuadric(m_pSphereObj);
    glDeleteTextures(1, &m_nTexID);
}

void vptz::Camera::GoToPosition(int x, int y) {
    lvDbgExceptionWatch;
    double tempHoriAngle = horiAngle;
    double tempVertiAngle = vertiAngle;
    PTZPointXYtoHV(x, y, horiAngle, vertiAngle, int(outputWidth), int(outputHeight), vertiFOV, horiAngle, vertiAngle);
    horiAngleChange = horiAngle-tempHoriAngle;
    if(horiAngleChange>180.0)
        horiAngleChange = 360.0-horiAngleChange;
    else if(horiAngleChange<=-180.0)
        horiAngleChange = horiAngleChange+360.0;
    vertiAngleChange = vertiAngle-tempVertiAngle;
}

void vptz::Camera::GoToPosition(cv::Point target) {
    lvDbgExceptionWatch;
    GoToPosition(target.x, target.y);
}

void vptz::Camera::BeginPlaying(int nFrameIdx) {
    lvDbgExceptionWatch;
    Set(PTZ_CAM_FRAME_POS,(double)nFrameIdx);
    currentTime = nFrameIdx/m_dFrameRate;
    firstTick = double(cv::getTickCount());
}

bool vptz::Camera::WaitDelay(bool bSleep) {
    lvDbgExceptionWatch;
    if(m_nCurrFrameIdx>=m_nScenarioFrameCount)
        return false;
    // The end of other parts (tracker's analysing, get frame)
    executionEndTime = (double(cv::getTickCount())-firstTick)/cv::getTickFrequency();
    // two of the three kinds of delay
    executionDelay = (executionEndTime-executionBeginTime)-getFrameOverhead;
    motionDelay = std::abs(horiAngleChange)/horiSpeed+std::abs(vertiAngleChange)/vertiSpeed;
    // calculate frame position
    currentTime += executionDelayRatio*executionDelay + motionDelay + communicationDelay;
    m_nCurrFrameIdx = std::max(int(currentTime*m_dFrameRate+0.5),m_nCurrFrameIdx+1);
    if(bSleep) {
        double waitingTime = motionDelay + communicationDelay - getFrameOverhead;
        if(waitingTime<0.001) waitingTime = 0.001;
        cv::waitKey(int(waitingTime*1000));
    }
    // the beginning of other parts (tracker's analysing, get frame)
    executionBeginTime = (double(cv::getTickCount())-firstTick)/cv::getTickFrequency();
    return m_nCurrFrameIdx<m_nScenarioFrameCount;
}

const cv::Mat& vptz::Camera::GetFrame() {
    lvDbgExceptionWatch;
    m_pContext->setAsActive();
    double duration = double(cv::getTickCount());
    if(isVideo) {
        if(!panoCapture.set(cv::CAP_PROP_POS_FRAMES,m_nCurrFrameIdx) || !panoCapture.read(panoImage))
            return panoImage;
    }
    if(panoImage.channels()==3)
        cv::cvtColor(panoImage,panoImage,cv::COLOR_BGR2BGRA);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(GLfloat(vertiAngle-180), 1, 0, 0);
    glRotatef(GLfloat(90-horiAngle), 0, 0, 1);
    glScalef(1.0, -1.0, -1.0);
    glRotatef(90.0, 0, 0, 1); // z axis, to make the center of the panoramic frame match the default camera direction (0, 90)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,panoImage.cols,panoImage.rows,0,GL_BGRA,GL_UNSIGNED_INT_8_8_8_8_REV,panoImage.data);
    gluSphere(m_pSphereObj, 5.0, sphereGridSize, sphereGridSize);    // generate the sphe
    glReadPixels(0,0,m_oViewportFrame.cols,m_oViewportFrame.rows,GL_BGRA,GL_UNSIGNED_INT_8_8_8_8_REV,m_oViewportFrame.data);
    cv::flip(m_oViewportFrame,m_oViewportFrame, 0);
    getFrameOverhead = (double(cv::getTickCount())-duration)/cv::getTickFrequency();
    glErrorCheck;
    return m_oViewportFrame;
}

void vptz::Camera::UpdatePanoImage(cv::Mat& image) {
    lvDbgExceptionWatch;
    lvAssert(!isVideo);
    image.copyTo(panoImage);
}

double vptz::Camera::Get(CameraPropertyFlag flag) {
    lvDbgExceptionWatch;
    switch(flag) {
        case PTZ_CAM_IS_VIDEO : return isVideo;
        case PTZ_CAM_FRAME_NUM : return m_nScenarioFrameCount;
        case PTZ_CAM_GET_FRAME_OVERHEAD : return getFrameOverhead;
        case PTZ_CAM_EXECUTION_DELAY : return executionDelay;
        case PTZ_CAM_MOTION_DELAY : return motionDelay;
        case PTZ_CAM_FRAME_RATE : return m_dFrameRate;
        case PTZ_CAM_FRAME_POS : return m_nCurrFrameIdx;
        case PTZ_CAM_VERTI_FOV : return vertiFOV;
        case PTZ_CAM_OUTPUT_WIDTH : return outputWidth;
        case PTZ_CAM_OUTPUT_HEIGHT : return outputHeight;
        case PTZ_CAM_HORI_ANGLE : return horiAngle;
        case PTZ_CAM_VERTI_ANGLE : return vertiAngle;
        case PTZ_CAM_HORI_SPEED : return horiSpeed;
        case PTZ_CAM_VERTI_SPEED : return vertiSpeed;
        case PTZ_CAM_EXECUTION_DELAY_RATIO : return executionDelayRatio;
        case PTZ_CAM_COMMUNICATION_DELAY : return communicationDelay;
        default : lvError("invalid flag");
    }
}

void vptz::Camera::Set(CameraPropertyFlag flag, double value) {
    lvDbgExceptionWatch;
    bool bRefreshProj = false;
    switch(flag) {
        case PTZ_CAM_FRAME_RATE:
            m_dFrameRate = value;
            break;
        case PTZ_CAM_FRAME_POS:
            if(value>m_nScenarioFrameCount-1||value<0)
                lvError("invalid framePos value");
            m_nCurrFrameIdx = (int)value;
            break;
        case PTZ_CAM_VERTI_FOV:
            if(value>=180.0 || value<=0.0)
                lvError("invalid vertiFOV value");
            vertiFOV = value;
            bRefreshProj = true;
            break;
        case PTZ_CAM_HORI_ANGLE: {
            if(value<=-180.0 || value>180.0)
                value = fmod(value+180-360*(value<0),360)-180+360*(value<0);
            double tempHoriAngle = horiAngle;
            horiAngle = value;
            horiAngleChange = horiAngle-tempHoriAngle;
            if(horiAngleChange>180.0)
                horiAngleChange = 360.0-horiAngleChange;
            else if(horiAngleChange<=-180.0)
                horiAngleChange = horiAngleChange+360.0;
        } break;
        case PTZ_CAM_VERTI_ANGLE: {
            if(value<0.0 || value>180.0)
                lvError("invalid vertiAngle value");
            double tempVertiAngle = vertiAngle;
            vertiAngle = value;
            vertiAngleChange = vertiAngle-tempVertiAngle;

        } break;
        case PTZ_CAM_HORI_SPEED:
            if(value<=0)
                lvError("invalid horiSpeed value");
            horiSpeed = value;
            break;
        case PTZ_CAM_VERTI_SPEED:
            if(value<=0)
                lvError("invalid vertiSpeed value");
            vertiSpeed = value;
            break;
        case PTZ_CAM_EXECUTION_DELAY_RATIO:
            if(value<0)
                lvError("invalid executionDelayRatio value");
            executionDelayRatio = value;
            break;
        case PTZ_CAM_COMMUNICATION_DELAY:
            if(value<0)
                lvError("invalid communicationDelay value");
            communicationDelay = value;
            break;
        default:
            lvError("invalid flag");
    }
    if(bRefreshProj && m_pContext) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(vertiFOV, outputWidth/outputHeight, 0.01, 10.0);
        glErrorCheck;
        glMatrixMode(GL_MODELVIEW);
    }
}

vptz::GTTranslator::GTTranslator( Camera* pCam, int bgt_output_width, int bgt_output_height, double bgt_verti_FOV) :
        m_pCamera(pCam) {
    lvDbgExceptionWatch;
    if(bgt_output_width<1 || bgt_output_height<1 || bgt_verti_FOV<=0.0 || bgt_verti_FOV>=180.0)
        lvError("invalid parameter(s)");
    curOutputWidth = int(m_pCamera->Get(PTZ_CAM_OUTPUT_WIDTH));
    curOutputHeight = int(m_pCamera->Get(PTZ_CAM_OUTPUT_HEIGHT));
    curHoriAngle = m_pCamera->Get(PTZ_CAM_HORI_ANGLE);
    curVertiAngle = m_pCamera->Get(PTZ_CAM_VERTI_ANGLE);
    curVertiFOV = m_pCamera->Get(PTZ_CAM_VERTI_FOV);
    bgtOutputWidth = bgt_output_width;
    bgtOutputHeight = bgt_output_height;
    bgtVertiFOV = bgt_verti_FOV;
}

vptz::GTTranslator::GTTranslator( int cur_output_width, int cur_output_height, double cur_verti_FOV,
                                  double cur_hori_angle, double cur_verti_angle, int bgt_output_width,
                                  int bgt_output_height, double bgt_verti_FOV) :
        m_pCamera(NULL) {
    lvDbgExceptionWatch;
    if(cur_output_width<1 || cur_output_height<1 || cur_verti_FOV<=0.0 || cur_verti_FOV>=180.0 ||
        bgt_output_width<1 || bgt_output_height<1 || bgt_verti_FOV<=0.0 || bgt_verti_FOV>=180.0 ||
        curHoriAngle<=-180.0 || curHoriAngle>180.0 || curVertiAngle<0.0 || curVertiAngle>180.0)
        lvError("invalid parameter(s)");
    curOutputWidth = cur_output_width;
    curOutputHeight = cur_output_height;
    curHoriAngle = cur_hori_angle;
    curVertiAngle = cur_verti_angle;
    curVertiFOV = cur_verti_FOV;
    bgtOutputWidth = bgt_output_width;
    bgtOutputHeight = bgt_output_height;
    bgtVertiFOV = bgt_verti_FOV;
}

void vptz::GTTranslator::UpdateViewAngle(double cur_hori_angle, double cur_verti_angle) {
    lvDbgExceptionWatch;
    if(curHoriAngle<=-180.0 || curHoriAngle>180.0 || curVertiAngle<0.0 || curVertiAngle>180.0)
        lvError("invalid input parameter");
    curHoriAngle = cur_hori_angle;
    curVertiAngle = cur_verti_angle;
}

bool vptz::GTTranslator::GetGTTargetPoint(double bgtHoriAngle, double bgtVertiAngle, cv::Point& tgtTargetPoint) {
    lvDbgExceptionWatch;
    if(bgtHoriAngle<=-180.0 || bgtHoriAngle>180.0 || bgtVertiAngle<0.0 || bgtVertiAngle>180.0)
        lvError("invalid input parameter");
    if(m_pCamera) {
        curHoriAngle = m_pCamera->Get(PTZ_CAM_HORI_ANGLE);
        curVertiAngle = m_pCamera->Get(PTZ_CAM_VERTI_ANGLE);
        curVertiFOV = m_pCamera->Get(PTZ_CAM_VERTI_FOV);
    }
    return PTZPointHVtoXY( bgtHoriAngle, bgtVertiAngle,
                           tgtTargetPoint.x, tgtTargetPoint.y,
                           curOutputWidth, curOutputHeight,
                           curVertiFOV, curHoriAngle, curVertiAngle);
}

bool vptz::GTTranslator::GetGTBoundingBox(double bgtHoriAngle, double bgtVertiAngle, int bgtBBoxWidth, int bgtBBoxHeight, cv::Rect& tgtBoundingBox) {
    lvDbgExceptionWatch;
    if(bgtHoriAngle<=-180.0 || bgtHoriAngle>180.0 || bgtVertiAngle<0.0 || bgtVertiAngle>180.0 ||
        bgtBBoxWidth<0 || bgtBBoxWidth>bgtOutputWidth || bgtBBoxHeight<0 || bgtBBoxHeight>bgtOutputHeight)
        lvError("VirtualPTZ Error: in GTTranslator::GetGTBoundingBox, invalid input parameter\n");
    if(m_pCamera) {
        curHoriAngle = m_pCamera->Get(PTZ_CAM_HORI_ANGLE);
        curVertiAngle = m_pCamera->Get(PTZ_CAM_VERTI_ANGLE);
        curVertiFOV = m_pCamera->Get(PTZ_CAM_VERTI_FOV);
    }
    // 1. Ground Truth Domain
    // the four vertexes of bounding box on image plane
    cv::Point l_t_XY((bgtOutputWidth-bgtBBoxWidth)/2, (bgtOutputHeight-bgtBBoxHeight)/2);
    cv::Point l_b_XY((bgtOutputWidth-bgtBBoxWidth)/2, (bgtOutputHeight+bgtBBoxHeight)/2);
    cv::Point r_t_XY((bgtOutputWidth+bgtBBoxWidth)/2, (bgtOutputHeight-bgtBBoxHeight)/2);
    cv::Point r_b_XY((bgtOutputWidth+bgtBBoxWidth)/2, (bgtOutputHeight+bgtBBoxHeight)/2);
    // the four vertexes of bounding box on spherical surface
    cv::Point2d l_t_HV, l_b_HV, r_t_HV, r_b_HV;
    PTZPointXYtoHV(l_t_XY, l_t_HV, bgtOutputWidth, bgtOutputHeight, bgtVertiFOV, bgtHoriAngle, bgtVertiAngle);
    PTZPointXYtoHV(l_b_XY, l_b_HV, bgtOutputWidth, bgtOutputHeight, bgtVertiFOV, bgtHoriAngle, bgtVertiAngle);
    PTZPointXYtoHV(r_t_XY, r_t_HV, bgtOutputWidth, bgtOutputHeight, bgtVertiFOV, bgtHoriAngle, bgtVertiAngle);
    PTZPointXYtoHV(r_b_XY, r_b_HV, bgtOutputWidth, bgtOutputHeight, bgtVertiFOV, bgtHoriAngle, bgtVertiAngle);
    // 2. Current Simulator Domain
    // the four vertexes of bounding box on image plane
    if( !PTZPointHVtoXY(l_t_HV, l_t_XY, curOutputWidth, curOutputHeight, curVertiFOV, curHoriAngle, curVertiAngle) ||
        !PTZPointHVtoXY(l_b_HV, l_b_XY, curOutputWidth, curOutputHeight, curVertiFOV, curHoriAngle, curVertiAngle) ||
        !PTZPointHVtoXY(r_t_HV, r_t_XY, curOutputWidth, curOutputHeight, curVertiFOV, curHoriAngle, curVertiAngle) ||
        !PTZPointHVtoXY(r_b_HV, r_b_XY, curOutputWidth, curOutputHeight, curVertiFOV, curHoriAngle, curVertiAngle))
        return false;
    // rectify
    cv::Point rectLeftTop((l_t_XY.x+l_b_XY.x)/2, (l_t_XY.y+r_t_XY.y)/2);
    cv::Point rectRightBottom((r_b_XY.x+r_t_XY.x)/2, (r_b_XY.y+l_b_XY.y)/2);
    tgtBoundingBox = cv::Rect(rectLeftTop, rectRightBottom);
    return true;
}

vptz::Evaluator::Evaluator( const std::string& sInputScenarioPath, const std::string& sInputGTSequencePath,
                            const std::string& sInputTargetMaskPath, const std::string& sOutputEvalFilePath,
                            double dCommDelay, double dExecDelayRatio, int nFirstTestFrameIdx, int nLastTestFrameIdx) :
        m_bUsingMultiTestSet(false) {
    lvDbgExceptionWatch;
    TestMetadata oDefaultTest = { "default_test",
                                  sInputScenarioPath,
                                  sInputGTSequencePath,
                                  sInputTargetMaskPath,
                                  nFirstTestFrameIdx,
                                  nLastTestFrameIdx};
    m_voTestSet.push_back(oDefaultTest);
    m_dCommDelay = dCommDelay;
    m_dExecDelayRatio = dExecDelayRatio;
    m_nCurrTestIdx = -1;
    m_oOutputEvalFS.open(sOutputEvalFilePath, cv::FileStorage::WRITE);
    if(!m_oOutputEvalFS.isOpened())
        lvError("cannot open the output yml file");
    m_bRunning = m_bReady = m_bQueried = false;
    m_dOutOfViewFrameRatio_FullAvg = 0;
    m_dProcessedFrameRatio_FullAvg = 0;
    m_dTargetPointError_FullAvg = 0;
    m_dTargetPointOffset_FullAvg = 0;
    m_dBBoxOverlapRatio_FullAvg = 0;
    m_dTrackFragmentation_FullAvg = 0;
    m_nTotSeqsTested = 0;
    m_nTotTestFrameCount = 0;
}

vptz::Evaluator::Evaluator( const std::string& sInputTestSetPath, const std::string& sOutputEvalFilePath,
                            double dCommDelay, double dExecDelayRatio) :
        m_bUsingMultiTestSet(true) {
    lvDbgExceptionWatch;
    cv::FileStorage oTestSet_FS(sInputTestSetPath,cv::FileStorage::READ);
    if(!oTestSet_FS.isOpened())
        lvError("cannot open the input test set file storage");
    cv::FileNode voTestSet_FN = oTestSet_FS["test_set"];
    if(voTestSet_FN.empty())
        lvError("test set file contains no test data");
    m_sDatasetRootPath = GetRootFolderPath(sInputTestSetPath)+"../";
    for(auto oTestIter=voTestSet_FN.begin(); oTestIter!=voTestSet_FN.end(); ++oTestIter) {
        TestMetadata oNewTest = { (*oTestIter)["test_name"],
                                  m_sDatasetRootPath+(std::string)((*oTestIter)["input_scenario_path"]),
                                  m_sDatasetRootPath+(std::string)((*oTestIter)["input_gtseq_path"]),
                                  m_sDatasetRootPath+(std::string)((*oTestIter)["input_target_mask_path"]),
                                  (*oTestIter)["init_frame_idx"],
                                  (*oTestIter)["last_frame_idx"]};
        m_voTestSet.push_back(oNewTest);
    }
    m_dCommDelay = dCommDelay;
    m_dExecDelayRatio = dExecDelayRatio;
    m_nCurrTestIdx = -1;
    m_oOutputEvalFS.open(sOutputEvalFilePath, cv::FileStorage::WRITE);
    if(!m_oOutputEvalFS.isOpened())
        lvError("cannot open the output yml file");
    m_oOutputEvalFS << "input_testset_path" << sInputTestSetPath;
    m_oOutputEvalFS << "sequences" << "[";
    m_bRunning = m_bReady = m_bQueried = false;
    m_dOutOfViewFrameRatio_FullAvg = 0;
    m_dProcessedFrameRatio_FullAvg = 0;
    m_dTargetPointError_FullAvg = 0;
    m_dTargetPointOffset_FullAvg = 0;
    m_dBBoxOverlapRatio_FullAvg = 0;
    m_dTrackFragmentation_FullAvg = 0;
    m_nTotSeqsTested = 0;
    m_nTotTestFrameCount = 0;
}

vptz::Evaluator::~Evaluator() {
    lvDbgExceptionWatch;
    if(std::uncaught_exception())
        return; // YML file is most likely in an irrecuperable state
    if(m_bUsingMultiTestSet) {
        if(m_bReady)
            m_oOutputEvalFS << "}";
        m_oOutputEvalFS << "]"
            << "targetPointError_fullavg" << m_dTargetPointError_FullAvg/m_nTotSeqsTested
            << "targetPointOffset_fullavg" << m_dTargetPointOffset_FullAvg/m_nTotSeqsTested
            << "bBoxOverlapRatio_fullavg" << m_dBBoxOverlapRatio_FullAvg/m_nTotSeqsTested
            << "trackFragmentation_fullavg" << m_dTrackFragmentation_FullAvg/m_nTotSeqsTested
            << "outOfViewRatio_fullavg" << m_dOutOfViewFrameRatio_FullAvg/m_nTotSeqsTested
            << "processedRatio_fullavg" << m_dProcessedFrameRatio_FullAvg/m_nTotSeqsTested
            << "tot_potential_frame_count" << m_nTotTestFrameCount;
    }
}

int vptz::Evaluator::GetTestSetSize() {
    return (int)m_voTestSet.size();
}

int vptz::Evaluator::GetCurrTestIdx() {
    return m_nCurrTestIdx;
}

void vptz::Evaluator::SetupTesting(int nTestIdx) {
    lvDbgExceptionWatch;
    lvAssert(nTestIdx<(int)m_voTestSet.size() && nTestIdx>=0);
    lvAssert(!m_bRunning);
    if(m_bUsingMultiTestSet) {
        if(m_bReady)
            m_oOutputEvalFS << "}";
        m_oOutputEvalFS << "{" << "id" << nTestIdx+1;
    }
    try {
        Setup( m_voTestSet[nTestIdx].sInputScenarioPath,
               m_voTestSet[nTestIdx].sInputGTSequencePath,
               m_voTestSet[nTestIdx].sInputTargetMaskPath,
               m_voTestSet[nTestIdx].nFirstTestFrameIdx,
               m_voTestSet[nTestIdx].nLastTestFrameIdx);
        m_pCamera->Set(PTZ_CAM_HORI_ANGLE,m_bgtInitHoriAngle);
        m_pCamera->Set(PTZ_CAM_VERTI_ANGLE,m_bgtInitVertiAngle);
    }
    catch(const vptz::Exception&) {
        if(m_bUsingMultiTestSet)
            m_oOutputEvalFS << "}";
        m_bReady = m_bQueried = false;
        m_nCurrTestIdx = -1;
        throw;
    }
    m_nCurrTestIdx = nTestIdx;
    m_bReady = true;
    m_bQueried = false;
    m_oCurrFrame = cv::Mat();
}

void vptz::Evaluator::BeginTesting() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    try {
        m_pCamera->BeginPlaying(m_nFirstTestFrameIdx);
    }
    catch(const vptz::Exception&) {
        if(m_bUsingMultiTestSet)
            m_oOutputEvalFS << "}";
        m_bReady = m_bQueried = false;
        m_nCurrTestIdx = -1;
        throw;
    }
    m_oOutputEvalFS << "results" << "[";
    m_nCurrOutOfViewFrameCount = 0;
    m_nCurrProcessedFrameCount = 0;
    m_dCurrTargetPointErrorSum = 0;
    m_dCurrTargetPointOffsetSum = 0;
    m_dCurrBBoxOverlapRatioSum = 0;
    m_bRunning = true;
    m_bQueried = false;
    m_oCurrFrame = cv::Mat();
}

void vptz::Evaluator::EndTesting() {
    lvDbgExceptionWatch;
    lvAssert(m_bRunning);
    const int nValidProcessedFrames = std::max(m_nCurrProcessedFrameCount-m_nCurrOutOfViewFrameCount,1);
    const double dTargetPointError = m_dCurrTargetPointErrorSum/nValidProcessedFrames;
    const double dTargetPointOffset = m_dCurrTargetPointOffsetSum/nValidProcessedFrames;
    const double dBBoxOverlapRatio = m_dCurrBBoxOverlapRatioSum/nValidProcessedFrames;
    const double dTrackFragmentation = double(m_nCurrOutOfViewFrameCount)/m_nCurrProcessedFrameCount;
    const double dOutOfViewRatio = (double)m_nCurrOutOfViewFrameCount/(m_nTestFrameCount-1);
    const double dProcessedRatio = (double)m_nCurrProcessedFrameCount/(m_nTestFrameCount-1);
    m_oOutputEvalFS << "]"
                    << "targetPointError_avg" << dTargetPointError
                    << "targetPointOffset_avg" << dTargetPointOffset
                    << "bBoxOverlapRatio_avg" << dBBoxOverlapRatio
                    << "trackFragmentation" << dTrackFragmentation
                    << "outOfViewRatio" << dOutOfViewRatio
                    << "processedRatio" << dProcessedRatio
                    << "potential_frame_count" << m_nTestFrameCount;
    if(m_bUsingMultiTestSet)
        m_oOutputEvalFS << "}";
    m_bRunning = m_bReady = m_bQueried = false;
    printf("Metrics for seq#%d [%s]:\n\tTPE: %f\n\tTPO: %f\n\tBBOR: %f\n\tTF:%f\n",m_nCurrTestIdx+1,m_voTestSet[m_nCurrTestIdx].sTestName.c_str(),dTargetPointError,dTargetPointOffset,dBBoxOverlapRatio,dTrackFragmentation);
    m_dOutOfViewFrameRatio_FullAvg += dOutOfViewRatio;
    m_dProcessedFrameRatio_FullAvg += dProcessedRatio;
    m_dTargetPointError_FullAvg += dTargetPointError;
    m_dTargetPointOffset_FullAvg += dTargetPointOffset;
    m_dBBoxOverlapRatio_FullAvg += dBBoxOverlapRatio;
    m_dTrackFragmentation_FullAvg += dTrackFragmentation;
    m_nCurrTestIdx = -1;
    m_oCurrFrame = cv::Mat();
    ++m_nTotSeqsTested;
    m_nTotTestFrameCount += m_nTestFrameCount;
}

double vptz::Evaluator::GetCurrCameraProperty(CameraPropertyFlag flag) {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return m_pCamera->Get(flag);
}

std::string vptz::Evaluator::GetCurrTestSequenceName() {
    lvDbgExceptionWatch;
    lvAssert(m_nCurrTestIdx<(int)m_voTestSet.size() && m_nCurrTestIdx>=0);
    return m_voTestSet[m_nCurrTestIdx].sTestName;
}

int vptz::Evaluator::GetPotentialTestFrameCount() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return m_nTestFrameCount;
}

cv::Mat vptz::Evaluator::GetInitTarget() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return m_oInitTarget;
}

cv::Mat vptz::Evaluator::GetInitTargetMask() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return m_oInitTargetMask;
}

cv::Mat vptz::Evaluator::GetInitTargetFrame() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return m_oInitTargetFrame;
}

cv::Rect vptz::Evaluator::GetInitTargetBBox() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_pCamera.get());
    return cv::Rect(m_oInitTargetFrame.cols/2-m_oInitTarget.cols/2,
                    m_oInitTargetFrame.rows/2-m_oInitTarget.rows/2,
                    m_oInitTarget.cols,m_oInitTarget.rows);
}

cv::Mat vptz::Evaluator::GetNextFrame(const cv::Point& oExpectedTargetPosition, bool bUseWaitDelaySleep) {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_bRunning && m_pCamera.get());
    m_pCamera->GoToPosition(oExpectedTargetPosition);
    if(!m_pCamera->WaitDelay(bUseWaitDelaySleep))
        m_oCurrFrame = cv::Mat();
    else {
        m_nCurrFrameIdx = int(m_pCamera->Get(PTZ_CAM_FRAME_POS));
        if(m_nCurrFrameIdx<m_nFirstTestFrameIdx || m_nCurrFrameIdx>m_nLastTestFrameIdx)
            m_oCurrFrame = cv::Mat();
        else
            m_oCurrFrame = m_pCamera->GetFrame().clone();
    }
    m_bQueried = !m_oCurrFrame.empty();
    return m_oCurrFrame;
}

cv::Mat vptz::Evaluator::GetNextFrame(double dExpectedTargetAngle_H, double dExpectedTargetAngle_V, bool bUseWaitDelaySleep) {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_bRunning && m_pCamera.get());
    m_pCamera->Set(PTZ_CAM_HORI_ANGLE,dExpectedTargetAngle_H);
    m_pCamera->Set(PTZ_CAM_VERTI_ANGLE,dExpectedTargetAngle_V);
    if(!m_pCamera->WaitDelay(bUseWaitDelaySleep))
        m_oCurrFrame = cv::Mat();
    else {
        m_nCurrFrameIdx = int(m_pCamera->Get(PTZ_CAM_FRAME_POS));
        if(m_nCurrFrameIdx<m_nFirstTestFrameIdx || m_nCurrFrameIdx>m_nLastTestFrameIdx)
            m_oCurrFrame = cv::Mat();
        else
            m_oCurrFrame = m_pCamera->GetFrame().clone();
    }
    m_bQueried = !m_oCurrFrame.empty();
    return m_oCurrFrame;
}

cv::Rect vptz::Evaluator::GetLastGTBoundingBox() {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_bRunning && m_pCamera.get());
    return tgtBoundingBox;
}

void vptz::Evaluator::UpdateCurrentResult(cv::Rect& rCurrTargetBBox, bool bPrintResult) {
    lvDbgExceptionWatch;
    lvAssert(m_bReady && m_bRunning && m_pCamera.get());
    lvAssert(m_bQueried);
    // basic ground truth for required frame
    int nCurrGTArrayIdx = m_nCurrFrameIdx-m_nFirstGTSeqFrameIdx;
    int bgtBBoxwidth = m_voCurrGTSequence_FN[nCurrGTArrayIdx]["width"];
    int bgtBBoxheight = m_voCurrGTSequence_FN[nCurrGTArrayIdx]["height"];
    double bgtHoriAngle = m_voCurrGTSequence_FN[nCurrGTArrayIdx]["horizontalAngle"];
    double bgtVertiAngle = m_voCurrGTSequence_FN[nCurrGTArrayIdx]["verticalAngle"];
    // translated ground truth for required frame
    m_pTranslator->GetGTTargetPoint(bgtHoriAngle, bgtVertiAngle, tgtTargetPoint);
    m_pTranslator->GetGTBoundingBox(bgtHoriAngle, bgtVertiAngle, bgtBBoxwidth, bgtBBoxheight, tgtBoundingBox);
    // target point distance and offset
    double dCurrTargetPointError;  // target point error (pixel, -1.0 if out of view), L2 dist between tracked point and gt point
    double dCurrTargetPointOffset; // target point offset (pixel, -1.0 if out of view), L2 dist between gt point and image center
    if(tgtTargetPoint.x<0 || tgtTargetPoint.y<0 ||
        tgtTargetPoint.x>m_pTranslator->curOutputWidth-1 || tgtTargetPoint.x>m_pTranslator->curOutputHeight-1 ||
        (rCurrTargetBBox.width<VPTZ_MINIMUM_BBOX_RADIUS || rCurrTargetBBox.height<VPTZ_MINIMUM_BBOX_RADIUS) ) {
        ++m_nCurrOutOfViewFrameCount;
        dCurrTargetPointError = -1.0;
        dCurrTargetPointOffset = -1.0;
    }
    else {
        cv::Point yourTargetPoint((rCurrTargetBBox.tl()+rCurrTargetBBox.br())*0.5);
        dCurrTargetPointError = norm(tgtTargetPoint-yourTargetPoint);
        m_dCurrTargetPointErrorSum += dCurrTargetPointError;
        cv::Point center(m_pTranslator->curOutputWidth/2, m_pTranslator->curOutputHeight/2);
        dCurrTargetPointOffset = norm(tgtTargetPoint-center);
        m_dCurrTargetPointOffsetSum += dCurrTargetPointOffset;
    }
    // overlap ratio of bounding box
    double AND = double((tgtBoundingBox&rCurrTargetBBox).area());
    double OR = double(tgtBoundingBox.area()+rCurrTargetBBox.area()-AND);
    double dCurrBBoxOverlapRatio = AND/OR ;
    m_dCurrBBoxOverlapRatioSum += dCurrBBoxOverlapRatio;
    m_oOutputEvalFS << "{:" << "framePos" << m_nCurrFrameIdx
                    << "targetPointError" << dCurrTargetPointError
                    << "targetPointOffset" << dCurrTargetPointOffset
                    << "boundingBoxOverlapRatio" << dCurrBBoxOverlapRatio << "}";
    if(bPrintResult)
        printf("Frame#%d:\n\tTPE: %f\n\tTPO: %f\n\tBBOR: %f\n",m_nCurrFrameIdx,dCurrTargetPointError,dCurrTargetPointOffset,dCurrBBoxOverlapRatio);
    ++m_nCurrProcessedFrameCount;
    m_bQueried = false;
}

void vptz::Evaluator::Setup( const std::string& sInputScenarioPath, const std::string& sInputGTSequencePath,
                             const std::string& sInputTargetMaskPath, int nFirstTestFrameIdx, int nLastTestFrameIdx) {
    lvDbgExceptionWatch;
    lvAssert(nFirstTestFrameIdx<nLastTestFrameIdx && nLastTestFrameIdx>0);
    m_bReady = m_bRunning = m_bQueried = false;
    m_pCamera.reset();
    m_pCamera = std::unique_ptr<Camera>(new Camera(sInputScenarioPath));
    m_pCamera->Set(PTZ_CAM_COMMUNICATION_DELAY,m_dCommDelay);
    m_pCamera->Set(PTZ_CAM_EXECUTION_DELAY_RATIO,m_dExecDelayRatio);
    m_oCurrGTSequence_FS.open(sInputGTSequencePath,cv::FileStorage::READ);
    if(!m_oCurrGTSequence_FS.isOpened())
        lvError("cannot open the input gt file");
    m_oOutputEvalFS << "input_filePath" << m_pCamera->m_sInputPath;
    m_nScenarioFrameCount = m_oCurrGTSequence_FS["totalFrameNum"];
    m_oOutputEvalFS << "input_totalFrameNum" << m_nScenarioFrameCount;
    int bgtFrameWidth = m_oCurrGTSequence_FS["frameImageWidth"];
    m_oOutputEvalFS << "input_frameImageWidth" << bgtFrameWidth;
    int bgtFrameHeight = m_oCurrGTSequence_FS["frameImageHeight"];
    m_oOutputEvalFS << "input_frameImageHeight" << bgtFrameHeight;
    double bgtVerticalFOV = m_oCurrGTSequence_FS["verticalFOV"];
    m_oOutputEvalFS << "input_verticalFOV" << bgtVerticalFOV;
    m_oOutputEvalFS << "input_execDelayRatio" << m_dExecDelayRatio;
    m_oOutputEvalFS << "input_commDelay" << m_dCommDelay;
    if(m_nScenarioFrameCount==0 || bgtFrameWidth==0 || bgtFrameHeight==0 || bgtVerticalFOV==0.0)
        lvError("invalid parameter(s) in input file");
    else if(m_nScenarioFrameCount != int(m_pCamera->Get(PTZ_CAM_FRAME_NUM)))
        lvError("frame count saved in ground truth file and panoramic video conflict");
    m_voCurrGTSequence_FN = m_oCurrGTSequence_FS["basicGroundTruth"];
    if(m_voCurrGTSequence_FN.empty())
        lvError("ground truth file contains no frame data");
    m_nFirstGTSeqFrameIdx = m_voCurrGTSequence_FN[(int)0]["framePos"];
    int nGTSeqFrameCount = (int)(m_voCurrGTSequence_FN.size());
    m_nLastTestFrameIdx = m_voCurrGTSequence_FN[nGTSeqFrameCount-1]["framePos"];
    if(m_nFirstGTSeqFrameIdx>m_nScenarioFrameCount)
        lvError("ground truth file contains no usable frame data for the current sequence");
    if(nGTSeqFrameCount!=m_nLastTestFrameIdx-m_nFirstGTSeqFrameIdx+1)
        lvError("ground truth file contains non-continuous frame annotations");
    if(nLastTestFrameIdx<m_nLastTestFrameIdx)
        m_nLastTestFrameIdx = nLastTestFrameIdx;
    m_nFirstTestFrameIdx = nFirstTestFrameIdx;
    if(m_nFirstTestFrameIdx<0)
        m_nFirstTestFrameIdx = m_nFirstGTSeqFrameIdx;
    else if(m_nFirstTestFrameIdx<m_nFirstGTSeqFrameIdx)
        lvError("invalid first test frame pos (GT starts later)");
    else if(m_nFirstTestFrameIdx>=m_nLastTestFrameIdx)
        lvError("invalid first test frame pos (GT ends before)");
    m_nTestFrameCount = m_nLastTestFrameIdx-m_nFirstTestFrameIdx;
    m_oOutputEvalFS << "input_first_test_frame_idx" << m_nFirstTestFrameIdx;
    m_oOutputEvalFS << "input_last_test_frame_idx" << m_nLastTestFrameIdx;
    int nInitGTArrayIdx = m_nFirstTestFrameIdx-m_nFirstGTSeqFrameIdx;
    m_bgtInitBBoxwidth = m_voCurrGTSequence_FN[nInitGTArrayIdx]["width"];
    m_bgtInitBBoxheight = m_voCurrGTSequence_FN[nInitGTArrayIdx]["height"];
    m_bgtInitHoriAngle = m_voCurrGTSequence_FN[nInitGTArrayIdx]["horizontalAngle"];
    m_bgtInitVertiAngle = m_voCurrGTSequence_FN[nInitGTArrayIdx]["verticalAngle"];
    m_pCamera->Set(PTZ_CAM_FRAME_POS,m_nFirstTestFrameIdx);
    m_pCamera->Set(PTZ_CAM_HORI_ANGLE,m_bgtInitHoriAngle);
    m_pCamera->Set(PTZ_CAM_VERTI_ANGLE,m_bgtInitVertiAngle);
    m_pTranslator = std::unique_ptr<GTTranslator>(new GTTranslator(m_pCamera.get(), bgtFrameWidth, bgtFrameHeight, bgtVerticalFOV));
    m_pTranslator->GetGTBoundingBox(m_bgtInitHoriAngle,m_bgtInitVertiAngle,m_bgtInitBBoxwidth,m_bgtInitBBoxheight,m_tgtInitBBox);
    m_oInitTargetFrame = m_pCamera->GetFrame().clone();
    m_oInitTarget = m_oInitTargetFrame(m_tgtInitBBox).clone();
    m_oInitTargetMask = cv::imread(sInputTargetMaskPath);
    if(!m_oInitTargetMask.empty()) {
        lvAssert(m_oInitTargetMask.size()==m_oInitTarget.size());
        std::vector<cv::Mat> oInitTargetMaskChannels;
        cv::split(m_oInitTargetMask,oInitTargetMaskChannels);
        m_oInitTargetMask = oInitTargetMaskChannels[0]>0;
        for(size_t nCh=1; nCh<oInitTargetMaskChannels.size(); ++nCh)
            m_oInitTargetMask |= oInitTargetMaskChannels[nCh]>0;
    }
    m_nCurrFrameIdx = m_nFirstTestFrameIdx;
    m_bgtBBoxwidth = m_bgtInitBBoxwidth;
    m_bgtBBoxheight = m_bgtInitBBoxheight;
    m_bgtHoriAngle = m_bgtInitHoriAngle;
    m_bgtVertiAngle = m_bgtInitVertiAngle;
}

void vptz::PTZPointXYtoHV( int target2dX, int target2dY, double& tarHoriAngle, double& tarVertiAngle,
                           int camOutputWidth, int camOutputHeight,
                           double camVertiFOV, double camHoriAngle, double camVertiAngle) {
    lvDbgExceptionWatch;
    if(camOutputWidth<1)
        lvError("invalid camOutputWidth");
    else if(camOutputHeight<1)
        lvError("invalid camOutputHeight");
    else if(target2dX<0 || target2dX>=camOutputWidth)
        lvError("invalid target2dX");
    else if(target2dY<0 || target2dY>=camOutputHeight)
        lvError("invalid target2dY");
    else if(camVertiFOV<=0.0 || camVertiFOV>=180.0)
        lvError("invalid camVertiFOV");
    else if(camHoriAngle<=-180.0 || camHoriAngle>180.0)
        lvError("invalid camHoriAngle");
    else if(camVertiAngle<0.0 || camVertiAngle>180.0)
        lvError("invalid camVertiAngle");

    // viewMatrix (world coordinate -> camera coordinate)
    // glRotatef(GLfloat(vertiAngle-180), 1, 0, 0);      // x axis
    double angle = D2R(camVertiAngle-180);
    cv::Mat rotateX = cv::Mat::eye(3, 3, CV_64FC1);
    rotateX.at<double>(1,1) = cos(angle);
    rotateX.at<double>(1,2) = -sin(angle);
    rotateX.at<double>(2,1) = sin(angle);
    rotateX.at<double>(2,2) = cos(angle);
    // glRotatef(GLfloat(90-horiAngle), 0, 0, 1);        // z axis
    cv::Mat rotateZ = cv::Mat::eye(3, 3, CV_64FC1);
    angle = D2R(90-camHoriAngle);
    rotateZ.at<double>(0,0) = cos(angle);
    rotateZ.at<double>(0,1) = -sin(angle);
    rotateZ.at<double>(1,0) = sin(angle);
    rotateZ.at<double>(1,1) = cos(angle);
    // final
    cv::Mat viewMatrix = rotateX*rotateZ;

    // 3d camera coordinates
    cv::Mat ray_camera(3, 1, CV_64F);
    ray_camera.at<double>(0,0) = target2dX-camOutputWidth/2.0;
    ray_camera.at<double>(1,0) = -(target2dY-camOutputHeight/2.0);
    ray_camera.at<double>(2,0) = -camOutputHeight/2.0/tan(D2R(camVertiFOV)/2.0);

    // 3d world coordinate
    cv::Mat ray_world = viewMatrix.inv()*ray_camera;

    // from 3d position to horizontal & vertical angles
    double target3dX = ray_world.at<double>(0,0);
    double target3dY = ray_world.at<double>(1,0);
    double target3dZ = ray_world.at<double>(2,0);
    tarHoriAngle = R2D(atan2(target3dY, target3dX));     // horizontal angle
    double target3dR = sqrt(target3dX*target3dX+target3dY*target3dY+target3dZ*target3dZ);
    tarVertiAngle = R2D(acos(target3dZ/target3dR));    // vertical angle
}

void vptz::PTZPointXYtoHV( cv::Point2i targetXY, cv::Point2d& targetHV, int camOutputWidth, int camOutputHeight,
                           double camVertiFOV, double camHoriAngle, double camVertiAngle) {
    lvDbgExceptionWatch;
    PTZPointXYtoHV(targetXY.x,targetXY.y,targetHV.x,targetHV.y, camOutputWidth, camOutputHeight, camVertiFOV, camHoriAngle, camVertiAngle);
}

bool vptz::PTZPointHVtoXY( double tarHoriAngle, double tarVertiAngle, int& target2dX, int& target2dY,
                           int camOutputWidth, int camOutputHeight,
                           double camVertiFOV, double camHoriAngle, double camVertiAngle) {
    lvDbgExceptionWatch;
    if(camOutputWidth<1)
        lvError("invalid camOutputWidth");
    else if(camOutputHeight<1)
        lvError("invalid camOutputHeight");
    else if(tarHoriAngle<=-180.0 || tarHoriAngle>180.0)
        lvError("invalid tarHoriAngle");
    else if(tarVertiAngle<0.0 || tarVertiAngle>180.0)
        lvError("invalid tarVertiAngle");
    else if(camVertiFOV<=0.0 || camVertiFOV>=180.0)
        lvError("invalid camVertiFOV");
    else if(camHoriAngle<=-180.0 || camHoriAngle>180.0)
        lvError("invalid camHoriAngle");
    else if(camVertiAngle<0.0 || camVertiAngle>180.0)
        lvError("invalid camVertiAngle");

    // If the direct difference between direction of target and camera is larger than 90 degrees
    cv::Point3d targetVec3d;
    targetVec3d.x = sin(D2R(tarVertiAngle))*cos(D2R(tarHoriAngle));    // *1.0 (unit vector)
    targetVec3d.y = sin(D2R(tarVertiAngle))*sin(D2R(tarHoriAngle));
    targetVec3d.z = cos(D2R(tarVertiAngle));
    cv::Point3d cameraVec3d;
    cameraVec3d.x = sin(D2R(camVertiAngle))*cos(D2R(camHoriAngle));
    cameraVec3d.y = sin(D2R(camVertiAngle))*sin(D2R(camHoriAngle));
    cameraVec3d.z = cos(D2R(camVertiAngle));
    double angleDifference = R2D(acos(targetVec3d.ddot(cameraVec3d)/(targetVec3d.ddot(targetVec3d)*cameraVec3d.ddot(cameraVec3d))));
    if(angleDifference>90.0) {
        target2dX = INT_MAX;
        target2dY = INT_MAX;
        return false;
    }

    // viewMatrix (world coordinate -> camera coordinate)
    // glRotatef(GLfloat(vertiAngle-180), 1, 0, 0);    // x axis
    double angle = D2R(camVertiAngle-180);
    cv::Mat rotateX = cv::Mat::eye(3, 3, CV_64FC1);
    rotateX.at<double>(1,1) = cos(angle);
    rotateX.at<double>(1,2) = -sin(angle);
    rotateX.at<double>(2,1) = sin(angle);
    rotateX.at<double>(2,2) = cos(angle);
    // glRotatef(GLfloat(90-horiAngle), 0, 0, 1);        // z axis
    cv::Mat rotateZ = cv::Mat::eye(3, 3, CV_64FC1);
    angle = D2R(90-camHoriAngle);
    rotateZ.at<double>(0,0) = cos(angle);
    rotateZ.at<double>(0,1) = -sin(angle);
    rotateZ.at<double>(1,0) = sin(angle);
    rotateZ.at<double>(1,1) = cos(angle);
    // final
    cv::Mat viewMatrix = rotateX*rotateZ;

    // 3d world coordinate: from horizontal & vertical angles to 3d position
    cv::Mat ray_world(3, 1, CV_64F);
    double vAngle = D2R(tarVertiAngle);
    double hAngle = D2R(tarHoriAngle);
    ray_world.at<double>(0,0) = sin(vAngle)*cos(hAngle);    // *1.0 (on unit sphere)
    ray_world.at<double>(1,0) = sin(vAngle)*sin(hAngle);
    ray_world.at<double>(2,0) = cos(vAngle);
    //cout<<"ray_world"<<ray_world<<endl;

    // 3d camera coordinates
    cv::Mat ray_camera = viewMatrix*ray_world;
    //cout<<"ray_camera"<<ray_camera<<endl;

    // 2d normalized coordinate, range [-0.5:0.5, -0.5:0.5]
    double x_camera = ray_camera.at<double>(0,0);
    double y_camera = ray_camera.at<double>(1,0);
    double z_camera = ray_camera.at<double>(2,0);
    double radio = (double(camOutputWidth)/double(camOutputHeight));
    double x_normal = x_camera/(-z_camera)/(radio*2.0*tan(D2R(camVertiFOV)/2.0));
    double y_normal = y_camera/(-z_camera)/(2.0*tan(D2R(camVertiFOV)/2.0));
    //cout<<cv::Point2d(x_normal, y_normal)<<endl;

    // 2d image coordinate, range [0:width, height:0]
    target2dX = int((x_normal+0.5)*camOutputWidth+0.5);
    target2dY = int((0.5-y_normal)*camOutputHeight+0.5);
    return true;
}

bool vptz::PTZPointHVtoXY( cv::Point2d targetHV, cv::Point2i& targetXY, int camOutputWidth, int camOutputHeight,
                           double camVertiFOV, double camHoriAngle, double camVertiAngle) {
    lvDbgExceptionWatch;
    return PTZPointHVtoXY(targetHV.x, targetHV.y, targetXY.x, targetXY.y, camOutputWidth, camOutputHeight, camVertiFOV, camHoriAngle, camVertiAngle);
}
