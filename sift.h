//#ifndef __SIFT_DESCRIPTOR_HH
//#define __SIFT_DESCRIPTOR_HH

#include <cv.h>
#include <stdio.h>

// beginnings of translation to C

typedef CvMatND CvSiftDescriptor;

CvSiftDescriptor * cvCreateSiftDescriptor(int width=2, int height=4, int rbins=8){
	int sizes[3] = {width, height, rbins};
	return cvCreateMatND(3, sizes, CV_32FC1);
}
float cvSiftGet(CvSiftDescriptor * m, int i, int j, int b){
	return *(float*)(m->data.ptr+i*m->dim[0].step+j*m->dim[1].step+b*m->dim[2].step);
}

int main(){
	CvSiftDescriptor *d = cvCreateSiftDescriptor();
	for(int i=0; i<d->dim[0].size; i++){
		for(int j=0; j<d->dim[1].size; j++){
			for(int k=0; k<d->dim[2].size; k++){
				cvSiftGet(d, i,j,k);
			}
		}
	}
}

struct CvScalePoint {
	CvPoint2D32f x;
	int pyr_level;
	int subLevel;
	float scale;
	float imgScale;
	float angle;
};

void cvCalcGaussianOctave( const CvArr * src, const CvArr ** dst, int len, int use_overlap=0 ){
	float sigma=1.6;
	float s;
	int i;
	if(use_overlap){
		len -= 3;
	}
	if(len<=0){
		return;
	}
	for(i=0; i<len; i++){
		s = sigma * exp2(i/(double)len); // s * 2^i/len
		cvSmooth( src, dst[i], CV_GAUSSIAN, 0, 0, s );
	}
	if(use_overlap){
		for(i=0; i<3; i++){
		s = sigma * exp2((i+len)/(double)len);
		cvSmooth( src, dst[i], CV_GAUSSIAN, 0, 0, s );
	}
}

// here gauss_octave is len+1 and dog_octave is len
void cvCalcDogOctave( const CvArr ** gauss_octave, const CvArr ** dog_octave, int len){
	int i=0;
	for(i=0; i<len; i++){
		cvSub(gauss_octave[i+1], gauss_octave[i], dog_octave[i]);
	}
}

void icvDoGFeatureDetectOctave( const CvArr ** gauss_octave, const CvArr ** dog_octave, int len){
	CvMat *min1, *min2, *min3, *max1, *max2, *max3, *tmp;
	cvIsLocalMin( dog_octave[0], min1 );
	cvIsLocalMax( dog_octave[0], max1 );
	cvIsLocalMin( dog_octave[1], min2 );
	cvIsLocalMax( dog_octave[1], max2 );
	for(int i=2; i<len; i++){
		cvIsLocalMin( dog_octave[i], min3 );
		cvIsLocalMax( dog_octave[i], max3 );
		
		CV_SWAP(min1,min2,tmp);
		CV_SWAP(min2,min3,tmp);
		CV_SWAP(max1,max2,tmp);
		CV_SWAP(max2,max3,tmp);
	}
}

void cvDoGFeatureDetect( const CvArr * im, CvMemStorage * storage, CvSeq ** features, 
		float dogThresh=3.0, float edgeRatio=5.5, float contrastThresh=0.03, float sigma=1.6,    
		int minImageSize=16, int nsub=1 ){
}

#if 0
	/*void calcGradient(const int &x, const int &y, const cv::Image<float> & space, 
	                  float & magnitude, float & angle){
		// compute gradient magnitude, orientation
		float dx = space[y][x+1]-space[y][x-1];
		float dy = space[y+1][x]-space[y-1][x];
		magnitude = sqrt( dx*dx + dy*dy );
		angle = atan2(dy,dx);
	}*/
	
	// Ix and Iy are pre rotated to orientation of point
	// feature is assumed to be located at the center of image patch
	template <typename image_t>
	void calcFromPatch(const ScalePoint &p, const image_t & Ix, const image_t & Iy) {
		(*this)=0.0;
		int n = this->length();
		int radius = n*size;
		int xmin = 0;//MAX(0, this->x-radius) - this->x + Ix.width/2;
		int ymin = 0;//MAX(0, this->y-radius) - this->y + Ix.height/2;
		int xmax = Ix.width;//MIN(width-1, this->x+radius) - this->x + Ix.width/2;
		int ymax = Ix.height;//MIN(height-1, this->y+radius) - this->y + Ix.height/2;
		for(int j=ymin; j<ymax; j++){
			int ybin = (j*size)/Ix.height;
			for(int i=xmin; i<xmax; i++){
				int xbin = (i*size)/Ix.width;
				
				float magnitude;
				float angle;
				ScalePoint::calcGradient(i,j,Ix,Iy,magnitude,angle);
				(*this)(xbin,ybin,ScalePoint::binGradient(angle,nbins)) += magnitude;
			}
		}
		this->normalize();
	}
	

	void calc(ScalePoint &p, const cv::DoGPyramid<float> & dog) {
		calc(p, dog.getIx(p.level, p.subLevel), dog.getIy(p.level, p.subLevel));
	}
	
	// Ix and Iy are standard sobel images from the pyramid, not preprocessed
	template <typename image_t>
	void calc(ScalePoint &p, const image_t & Ix, const image_t & Iy){
		(*this)=0.0;
		int n = this->length();
		int radius = n*size;
		int xmin = (int) (MAX(0, cvFloor(p.x)-radius));
		int ymin = (int) (MAX(0, cvFloor(p.y)-radius));
		int xmax = (int) (MIN(Ix.width, cvFloor(p.x)+radius));
		int ymax = (int) (MIN(Ix.height, cvFloor(p.y)+radius));
		int xoff = (int) (p.x-radius);
		int yoff = (int) (p.y-radius);
		for(int j=ymin; j<ymax; j++){
			int ybin = ((j-yoff)*size)/(radius*2);
			for(int i=xmin; i<xmax; i++){
				int xbin = ((i-xoff)*size)/(radius*2);

				float magnitude;
				float angle;
				ScalePoint::calcGradient(i,j,Ix,Iy,magnitude,angle);
				// need to account for angle relative to orientation
				angle-=p.angle;
				assert(xbin>=0 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(xbin<4 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(ybin>=0 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(ybin<4 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				(*this)(xbin,ybin,ScalePoint::binGradient(angle,nbins)) += magnitude;
			}
		}
		this->normalize();
	}
};

} // namespace sift
#endif

//#endif //__SIFT_DESCRIPTOR_H
