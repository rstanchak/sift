#ifndef __SIFT_CORRESPOND_HH
#define __SIFT_CORRESPOND_HH

#include <cv/sift_feature_detector.hh>

namespace sift {
	typedef cv::Image<float> image_t;
	typedef std::vector<ScalePoint> pvector_t;
	typedef std::vector<sift::Descriptor<> > dvector_t;

/** Finds corresponding points between im1 and im2 using SIFT features
 * @param im1 grey scale floating point image (scaled 0 to 256)
 * @param im2 grey scale floating point image (scaled 0 to 256)
 * @param pts1 output vector of ScalePoint (input should be size 0)
 * @param pts2 output vector of ScalePoint (input should be size 0)
 * @param keys1 output vector of sift::Descriptor (input size should be 0)
 * @param keys2 output vector of sift::Descriptor (input size should be 0)
 */
int correspond(const image_t & im1, const image_t & im2,
                    pvector_t & pts1, pvector_t & pts2,
				    dvector_t * keys1 = NULL, dvector_t * keys2 = NULL){
	sift::FeatureTracker detector(0.06);
	std::vector<ScalePoint> features;
	std::vector<sift::Descriptor<> > descriptors;

	// do feature detection on first image
	detector.track(im1);
	
	// copy points and descriptors
	features.insert(features.end(), detector.begin(), detector.end());
	descriptors.insert(descriptors.end(), detector.descriptors().begin(), detector.descriptors().end());

	detector.track(im2);

	for(int i=0; i<features.size(); i++){
		sift::FeatureTracker::iterator it = detector.labels()[i];
		if(it!= detector.end()){
			pts1.push_back( features[i] );
			pts2.push_back( *it );
			//if(keys1!=NULL)	keys1->push_back( descriptors[ i ] );
			//if(keys2!=NULL)	keys2->push_back( detector.descriptors()[i] );
		}	
	}
	return pts1.size();
}

} // namespace sift

#endif // __SIFT_CORRESPOND
