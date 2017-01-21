#include <cv/dog_scale_space.hh>

int main(int argc, char ** argv){
	cv::Image<float> im;
	cv::DoGPyramid<float> space(16,3);
	for(int k=1; k<argc; k++){
		im.open(argv[k]);
		space.calculate(im);
		cvNamedWindow("win",0);
		int i=3;
		int j=1;
		//for(int i=0; i<space.getNumScales()-1; i++){
		//	for(int j=1; j<space.getNumSubScales()-1; j++){
				im = space.getImage(i,j);
				cvAbs(&(space.getImage(i,j)), &im);
				im.convert().show("win");
				std::cout<<i<<","<<j<<" -- "<<"Sigma = "<<space.getSigma(i,j)<<std::endl;
				cvWaitKey(1000);
		//	}
		//}
	}
}
