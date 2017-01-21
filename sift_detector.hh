#ifndef __SIFT_DETECTOR_HH
#define __SIFT_DETECTOR_HH

class SIFTDetector {
	DoGScaleSpace _spaces;

	SIFTDetector(int octaves, int scales) : 
		_spaces(octaves, scales)
	{
	}
	
	void update(cv::Image<uchar> & in){
	}
	
	// retrieve feature, descriptor
		
};

#endif //__SIFT_DETECTOR_HH
