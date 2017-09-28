#include "litiv/vptz/virtualptzExtend.hpp"
cv::Mat vptz::EvaluatorEx::GetPanoramicFrame()
{
	return this->m_pCamera->panoImage;
}
cv::Point vptz::EvaluatorEx::GetPanoramicTargetPoint()
{
	return this->m_tgtCenterPoint;
}
double vptz::EvaluatorEx::GetSphereRadius()
{
	this->m_SephereRadius = this->m_pCamera->panoImage.cols / (2 * M_PI);//=constanct
	return this->m_SephereRadius;
}
double vptz::EvaluatorEx::GetTgtSpeedX()
{
#if meanVelocity
	return (this->m_tgtSpeed_x+this->m_oldTgtSpeed_x)/2;
#else
	return this->m_tgtSpeed_x;
#endif // meanVelocity

}
double vptz::EvaluatorEx::GetTgtSpeedY()
{
	
#if meanVelocity
	return (this->m_tgtSpeed_y+this->m_oldTgtSpeed_y)/2;
#else
	return this->m_tgtSpeed_y;
#endif // meanVelocity

}
double vptz::EvaluatorEx::GetAccelerationX()
{
	return this->m_acceleration_x;
}
double vptz::EvaluatorEx::GetAccelerationY()
{
	return this->m_acceleration_y;
}
void vptz::EvaluatorEx::UpdateCurrentPanoramicStatus()
{
	this->m_tgtCenterPoint= this->GetPanoPoint(cv::Point(this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_WIDTH) / 2, this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_HEIGHT) / 2));
	this->m_executionTime = this->GetCurrCameraProperty(PTZ_CAM_EXECUTION_DELAY) + this->GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY);
	double m_distance_x = this->GetPanoramicTargetPoint().x - this->m_oldTgtCenterPoint.x;
	double m_distance_y = this->GetPanoramicTargetPoint().y - this->m_oldTgtCenterPoint.y;
	if (this->m_executionTime != 0)
	{
		this->m_oldTgtSpeed_x = this->m_tgtSpeed_x;
		this->m_oldTgtSpeed_y = this->m_tgtSpeed_y;
		this->m_tgtSpeed_x = m_distance_x / this->m_executionTime;
		this->m_tgtSpeed_y = m_distance_y / this->m_executionTime;
	}
	//////////////////////////////////////////////////////////////////////////Update older center point in panoramic image
	this->m_oldTgtCenterPoint.x = m_tgtCenterPoint.x;
	this->m_oldTgtCenterPoint.y = m_tgtCenterPoint.y;


	//calculate acceleration using ancient, old and current point.
	if (this->m_executionTime != 0)
	{
		this->m_acceleration_x = (this->m_tgtSpeed_x - this->m_oldTgtSpeed_x) / m_executionTime;
		this->m_acceleration_y = (this->m_tgtSpeed_y - this->m_oldTgtSpeed_y) / m_executionTime;
	}
}

void vptz::EvaluatorEx::SetTrackingResult(bool result)
{
	this->m_tempTrackingResult = result;
}

bool vptz::EvaluatorEx::UpdateSpeedAccordingToWhetherObjTurnAround(cv::Point oNewCenterPos)
{
	cv::Point oNewCenterPosPano = this->GetPanoPoint(oNewCenterPos);
	double temp1 = (oNewCenterPosPano.x - this->GetPanoramicTargetPoint().x)*this->GetTgtSpeedX();

	if (temp1 < 0)
	{
		this->m_tgtSpeed_x = 0;
		this->m_tgtSpeed_y = 0;
		this->m_oldTgtSpeed_x = 0;
		this->m_oldTgtSpeed_y =0;
		this->m_acceleration_x = 0;
		this->m_acceleration_y = 0;
		return true;
	}
	else
		return false;
}

bool vptz::EvaluatorEx::GetTrackingResult()
{
	return this->m_tempTrackingResult;
}
/*@brief get point coordinates in pano according to the current point coordinates in camara view.
*/
cv::Point vptz::EvaluatorEx::GetPanoPoint(cv::Point target)
{

	double horiAngle;//get horiAngle according to the current horiAngle and the point in camera image.
	double vertiAngle;
	if (target.y > 0)
	{
		target.y = -1;
	}
	else if (target.y > this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_WIDTH))
	{
		target.y = this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_WIDTH)-1;
	}
	if (target.x < 0)
	{
		target.x = 0;
	}
	else if (target.x > this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_HEIGHT))
	{
		target.x = this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_HEIGHT)-1;
	}
	PTZPointXYtoHV(target.x, target.y, horiAngle, vertiAngle, this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_HEIGHT), this->GetCurrCameraProperty(PTZ_CAM_OUTPUT_WIDTH), this->GetCurrCameraProperty(PTZ_CAM_VERTI_FOV), this->GetCurrCameraProperty(PTZ_CAM_HORI_ANGLE), this->GetCurrCameraProperty(PTZ_CAM_VERTI_ANGLE));
	cv::Point oNewCenterPosPano;
	double tgt_x = 0, tgt_y = 0;
	tgt_x += this->GetPanoramicFrame().cols / 2;
	tgt_x -= horiAngle*M_PI*this->GetSphereRadius() / 180;
	tgt_y += vertiAngle*M_PI*this->GetSphereRadius() / 180;//attention:although image is (3500,1750), but the y is not the arc length, and I do not know what that is. 
	oNewCenterPosPano.x = tgt_x;
	oNewCenterPosPano.y = tgt_y;
	return oNewCenterPosPano;


}
