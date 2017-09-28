#pragma once
#include "virtualptz.hpp"
#include <opencv2/tracking.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
//@brief: choose position predection method. 
#define meanVelocity 0
#define accelerationVelocity 1
#if meanVelocity
#undef accelerationVelocity
#endif // !meanVelocity

namespace vptz {
	class EvaluatorEx : public Evaluator
	{
		friend class Camera;
	public:
		EvaluatorEx(const std::string& sInputTestSetPath,
			const std::string& sOutputEvalFilePath,
			double dCommDelay = 0.5,
			double dExecDelayRatio = 0.0,
			double dPredictScale = 0.0) :Evaluator(sInputTestSetPath, sOutputEvalFilePath, dCommDelay, dExecDelayRatio,dPredictScale) {
		}
		//@brief: returns the panoramic image
		cv::Mat GetPanoramicFrame();
		//@brief: returns the center point of the target object in the panoramic image.
		cv::Point GetPanoramicTargetPoint();
		//@brief: returns the sphere radius(sphere means the panoramic image forms a sphere)
		double GetSphereRadius();
		//@brief: returns the speed of the moving object
		double GetTgtSpeedX();
		double GetTgtSpeedY();
		double GetAccelerationX();
		double GetAccelerationY();

		//@brief: return tracking result(some trackers' update() function will return bool value)
		bool GetTrackingResult();
		//@brief: set tracking result.
		void SetTrackingResult(bool result);
		//@brief get point coordinates in panoramic image according to the current point coordinates in camara view.
		cv::Point vptz::EvaluatorEx::GetPanoPoint(cv::Point target);
		//@brief: update tgtCenter , speed, process time, acceleration and offset.
		void UpdateCurrentPanoramicStatus();
		//@brief: detect whether the obj turn around and reset obj speed, return true if indeed turn around.
		bool UpdateSpeedAccordingToWhetherObjTurnAround(cv::Point oNewCenterPos);
	private:
		double m_SephereRadius;//sphere radius=circumference /(2*PI)=panoImage.cols /(2*PI)
		double m_executionTime;//=PTZ_CAM_EXECUTION_DELAY+PTZ_CAM_MOTION_DELAY
		cv::Point m_tgtCenterPoint;//tracker target coodinates in panoImage.
		cv::Point m_oldTgtCenterPoint;// tracker target coodinates in panoImage in last valid frame, used for calculate the speed of object.
		double m_tgtSpeed_x=0;//tracker target speed
		double m_tgtSpeed_y=0;//tracker target speed
		double m_oldTgtSpeed_x=0;//tracker target speed
		double m_oldTgtSpeed_y=0;//tracker target speed
		double m_acceleration_x;//acceleration in x.
		double m_acceleration_y;//acceleration in x.
		bool m_tempTrackingResult;//=tracking.update()
	};


}

