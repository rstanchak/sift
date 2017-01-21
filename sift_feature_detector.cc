#include <highgui.h>
#include <cv/dog_feature_detector.hh>
#include <cv/wincompat.h>
#include <cv/sift_descriptor.hh>
#include <find_nearest.hh>
#include <cv/cvext.h>

inline double dist(const sift::Descriptor<> &a, const sift::Descriptor<> &b){
	return a.dist(b);
}
/*inline double dist(const ScalePoint<sift::Descriptor<> > &a, 
		           const ScalePoint<sift::Descriptor<> > &b){
	return a.descriptor.dist(b.descriptor);
}*/

int main(int argc, char ** argv){

	cvRedirectError( cvSegReport );
	double epsilon = 10;
	DoGFeatureDetector detector(16,6);
	cv::Image<unsigned char, 3> im3;
	cv::Image<unsigned char> im;
	cv::Image<float> imf;
	cvNamedWindow("gt", 1);
	cvNamedWindow("lt", 1);
	cvNamedWindow("p", 1);
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);
	std::vector<sift::Descriptor<> > features;
	for(int i=1; i<argc; i++){
		//std::cout<<"Opening "<<argv[i]<<std::endl;
		assert( im.open(argv[i]) );
		assert( im3.open(argv[i]) );
		//std::cout<<"Convert to float"<<argv[i]<<std::endl;	
		imf = im;
		//std::cout<<"Detecting features"<<argv[i]<<std::endl;
		detector.detect(imf);	
		char buffer[256];
		// show sub scales
		/*for(int j=0; j<detector.getPyramid().getNumScales(); j++){
			for(int k=0; k<=detector.getPyramid().getNumSubScales(); k++){
				std::cerr<<"Showing "<<j<<","<<k<<std::endl;
				sprintf(buffer, "pyr_%d_%d.bmp", j, k);
				detector.getPyramid().getGaussian(j,k).convert(1, 0).save(buffer);
			}
		}*/
		
		/*for(std::list<DoGFeatureDetector<sift::Descriptor>::feature_t>::const_iterator it=detector.begin(); it!=detector.end(); ++it){
			std::cout<<*it<<std::endl;
			continue;
				Keypoint k;
				if(!filterEdge(dog.getImage(ar[i][j].level(), ar[i][j].subLevel()), ar[i][j].x(), ar[i][j].y(), -20.0)){
					std::cerr<<"Edge failed"<<std::endl;
					continue;
				}
				if(!subPixelLocalize(dog, ar[i][j], k)){
					std::cerr<<"localization failed\n"<<std::endl;
					continue;
				}
				if(!filterContrast(k)){
					std::cerr<<"contrast failed"<<std::endl;
					continue;
				}
				
				std::cout<<"["<<it->flX()<<","<<it->flY()<<std::endl;
				cvCircle(&im3, cvPoint(it->imX(), it->imY()), it->getRadius(), CV_RGB(255,0,0));
		}
		*/
		
		//continue;
		//im3.show("p");
		//cvWaitKey(1);
		std::cout<<"Found "<<detector.size()<<" unfiltered feature points\n"<<std::endl;
		std::cout<<"Now filtering"<<std::endl;
		detector.filter();
		std::cout<<"Found "<<detector.size()<<" filtered feature points\n"<<std::endl;
		/*for(DoGFeatureDetector<sift::Descriptor>::const_iterator it=detector.begin(); it!=detector.end(); ++it){
			it->descriptor.calc(*it, detector.getPyramid());
			//std::cout<<"["<<it->x<<","<<it->y<<" @ "<<it->scale<<"]"<<std::endl;
			if(it->level!=0) continue;
			cvCircle(&im3, cvPoint(it->imX(), it->imY()), (int) it->getRadius(), CV_RGB(0,255,0));
			cvLine(&im3, cvPoint(it->imX(), it->imY()), 
			             cvPoint((int) (it->flX()+it->getRadius()*cos(it->angle)), (int)(it->flY()+it->getRadius()*sin(it->angle))),
						 CV_RGB(0,255,0),1);
			im3.show("p");
			cvWaitKey(10);
		}
		cvWaitKey(-1);
		*/
		std::vector<sift::Descriptor<> > new_features;
		sift::Descriptor<> descriptor;
		std::cout<<"Known descriptors: "<<features.size()<<std::endl;
		std::cout<<"Matching Found w/ Known"<<std::endl;
		for(DoGFeatureDetector::const_iterator it=detector.begin(); it!=detector.end(); ++it){
			// calculate descriptor
			descriptor.calc(*it, detector.getPyramid());

			std::vector<sift::Descriptor<> >::iterator nearest_descriptor;
			char buf[256];
			if(features.size()!=0){
				nearest_descriptor = find_nearest(descriptor, features.begin(), features.end());

			}
			// no features found yet OR feature is not unique to the OLD set  
			if(features.size()==0 || new_features.size()==0 || dist(descriptor, *nearest_descriptor)>epsilon){
				new_features.push_back(descriptor);
				nearest_descriptor = new_features.end();
				/*assert(descriptor == new_features[ new_features.size()-1 ]);
				if(new_features.size()>1){
					assert(new_features[0]!=new_features[1]);	
				}*/
			}
			cvCircle(&im3, cvPoint(it->imX(), it->imY()), (int) it->getRadius(), CV_RGB(0,255,0));
			cvLine(&im3, cvPoint(it->imX(), it->imY()), 
					     cvPoint((int) (it->flX()+it->getRadius()*cos(it->angle)), 
						 (int)(it->flY()+it->getRadius()*sin(it->angle))), CV_RGB(0,255,0),1);
			//snprintf(buf, 256, "%d", );
			//std::cout<<"feature is index "<<idx<<" "<<buf<<std::endl;
			//cvPutText(&im3, buf, cvPoint(it->imX(), it->imY()), &font, CV_RGB(255,0,0)); 
		}
		features.insert(features.end(), new_features.begin(), new_features.end());
		im3.show("p");
		cvWaitKey(500);
																											  
															
			    
	/*	cvWaitKey(-1);
		for(int j=0; j<detector.getPyramid().getNumScales(); j++){
			for(int k=1; k<detector.getPyramid().getNumSubScales()-1; k++){
				im3 = detector.getPyramid().getImage(j,k).convert(0.5,128);
				for(std::list<detector::feature_t>::const_iterator it=detector.begin(); it!=detector.end(); ++it){
					if(it->level!=j || it->subLevel!=k) continue;
					cvCircle(&im3, cvPoint(it->x, it->y), 2, CV_RGB(0,255,0));
				}
				im3.show("p");
				cvWaitKey(1);
				cvWaitKey(-1);
			}
		}*/
		/*for(int i=0; i<dog.getScaleSpace().length(); i++){
		  dog.getScaleSpace().getImage(i).convert(1,0).show("w");
		  cvWaitKey(-1);
		  }
		  for(int i=0; i<dog.length(); i++){
		  dog.getImage(i).convert(.5,128).show("w");
		  cvWaitKey(-1);
		  }*/
	}
}
