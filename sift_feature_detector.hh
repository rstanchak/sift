#ifndef __SIFT_FEATURE_DETECTOR_HH
#define __SIFT_FEATURE_DETECTOR_HH
#include <cv/sift_descriptor.hh>
#include <cv/dog_feature_detector.hh>
#include <cv/find_nearest.hh>

inline double dist(const sift::Descriptor<> &a, const sift::Descriptor<> &b){
	return a.dist(b);
}

/*int find_nearest(const sift::sift::Descriptor &d, std::vector<sift::sift::Descriptor> & array, double epsilon, void * ar){
	std::vector<float> score( array.size(), 0.0f);
	int max_index=-1;
	double sum=0,mean=0;
	int ne=0;
	for(int k=2; k>=0; k--){
		sum=0;
		ne=0;
		for(int i=0; i<array.size(); i++){
			if(k!=2 && score[i] > mean){
				//std::cout<<"skipping "<<k<<","<<i<<": "<<score[i]<<">"<<mean<<std::endl;
				score[i] = 128;
				continue;
			}
			//std::cout<<"not skipping "<<k<<","<<i<<": "<<score[i]<<">"<<mean<<std::endl;
			score[i] = array[i].dist(d, k);
			sum+=score[i];
			ne++;
			if(k==0 && (max_index==-1 || score[i] < score[max_index])) max_index=i;
		}
		mean = sum/ne;
	}
	if(score[max_index]>epsilon) return -1;
	return max_index;
}*/
	 
namespace sift {
class FeatureTracker : public DoGFeatureDetector {
	float _epsilon;
	std::vector<sift::Descriptor<> > _descriptors;
	std::vector<iterator> _labels;
public:
	FeatureTracker(float epsilon=1600) : 
		DoGFeatureDetector(),
		_epsilon(epsilon)
	{
	}
	FeatureTracker(float epsilon,
			float dogThresh,
			float edgeRatio,
			float sigma,
			int minImageSize, int nsub):
		DoGFeatureDetector(dogThresh,edgeRatio,sigma,minImageSize,nsub),
		_epsilon(epsilon)
	{
	}
	void setEpsilon(float eps){
		_epsilon = eps;
	}
	const std::vector<iterator> & labels() const { return _labels; }
	const std::vector<sift::Descriptor<> > & descriptors() const { return _descriptors; }
	void clear(){
		_descriptors.clear();
		_labels.clear();
		_features.clear();
	}
	void track(const image_t & im){
		// find features
		this->detect(im);

		// filter out bad features 
		this->filter();

		// insert descriptors initially
		if(_descriptors.size()==0){
			_descriptors.resize(_features.size());

			// insert descriptors
			int i=0;
			iterator it = _features.begin(); 
			_labels.resize(_features.size());
			while(it!=_features.end()){
				_descriptors[i].calc(*it, this->getPyramid());
				_labels[i]=it;
				++it;
				++i;
			}
		}
		// descriptors already saved -- just associate them
		else{
			sift::Descriptor<> desc;
			_labels.clear();
			// point all labels to end of features
			_labels.resize(_descriptors.size(), _features.end());
			
			for(iterator it = _features.begin(); it!=_features.end(); ++it){
				//calc descriptor for this feature
				desc.calc(*it, this->getPyramid());

				// find best match
				int idx = find_nearest(desc, _descriptors, _epsilon, NULL);
				if(idx>=0){
					_labels[idx] = it;
				}
			}
		}
	}
	void draw(IplImage * im, CvScalar color){
		CvFont font;
		char buf[256];
		cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);
		size_t i=0;
		for(i = 0; i<_labels.size(); i++){
			if(_labels[i]==_features.end()) continue;
			iterator it = _labels[i];
			CvPoint p1 = cvPoint(it->imX(), it->imY());
			CvPoint p2 = cvPoint((int) (it->flX()+it->getRadius()*cos(it->angle)),
					(int)(it->flY()+it->getRadius()*sin(it->angle)));
			snprintf(buf, 256, "%ld", i);

			cvCircle(im, p1, (int) it->getRadius(), color);
			cvLine(im, p1, p2, color);
			cvPutText(im, buf, p1, &font, color);
			i++;
		}
	}

}; // class FeatureDetector

} // namespace sift

#endif //__SIFT_FEATURE_DETECTOR_HH
