#include <highgui.h>
#include <cv/sift_feature_detector.hh>
#include <cv/wincompat.h>
#include <cv/cvext.h>


int main(int argc, char ** argv){
	cvRedirectError( cvSegReport );
	sift::FeatureTracker detector;
	cv::Image<unsigned char, 3> im3;
	cv::Image<unsigned char> im;
	cv::Image<float> imf;
	cvNamedWindow("gt", 1);
	cvNamedWindow("lt", 1);
	cvNamedWindow("p", 1);
	for(int i=1; i<argc; i++){
		//std::cout<<"Opening "<<argv[i]<<std::endl;
		im.open(argv[i]);
		im3.open(argv[i]);
		//std::cout<<"Convert to float"<<argv[i]<<std::endl;	
		imf.convert(im);
		//std::cout<<"Detecting features"<<argv[i]<<std::endl;
		detector.track(imf);	
		std::cout<<"Found "<<detector.size()<<" filtered feature points\n"<<std::endl;
		detector.draw(&im3, CV_RGB(0,255,0));
		
		im3.show("p");
		cvWaitKey(100);
	}
	cvWaitKey();
}
