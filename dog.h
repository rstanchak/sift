#ifndef __CV_DOG_HH
#define __CV_DOG_HH

#include <cxcore.h>
#include <cv.h>

CV_INLINE void cvDog(const CvArr * src, CvArr * dest, CvArr * temp, int param1, int param2){
	cvSmooth(src, temp, CV_GAUSSIAN, param1, param2, 0);
	cvSmooth(src, dest, CV_GAUSSIAN, param1+2, param2+2, 0);
	cvAbsDiff(dest,temp,dest);
}

#endif //__CV_DOG_HH
