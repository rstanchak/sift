#ifndef __SIFT_DESCRIPTOR_HH
#define __SIFT_DESCRIPTOR_HH

#include "scale_point.hh"
#include "array.h"
#include "dog_scale_space.hh"

struct CvSIFTDescriptor {
	CvScalePoint2D32f point;
	CvMat hist;  // of size rbins*sbins*sbins
	int rbins; //# of radial bins
	int sbins; //# of spatial bins

};

struct CvScalePoint2D32f {
	CvPoint2D32f center; // in world coordinates
	float scale;
	float orientation;
};

struct CvScaleSpace {
	CvMat ** images;
	int nScales;
};

struct CvScaleSpacePyr {
	CvScaleSpace * levels;
	int nLevels;
};

CvPoint cvPointFromScale32f(CvScalePoint2D32f p, int pyrlevel=0){
	CvPoint q;
	return q;
}

CvSIFTDescriptor * cvCreateSIFTDescriptor(int sbins, int rbins){
	CvSIFTDescriptor * s;

	CV_CALL( s = (CvSIFTDescriptor*)cvAlloc( sizeof(CvSIFTDescriptor)));

	memset(&s->point, 0, sizeof(CvScalePoint2D32f));

	CV_CALL( cvInitMatHeader(&s->hist, sbins*sbins*rbins, 1, CV_32FC1) );
	CV_CALL( cvCreateData(&s->hist) );
	
	s->sbins = sbins;
	s->rbins = rbins;

	return s;
}

/** Allocate scale space data structure */
CvScaleSpacePyr * cvCreateScaleSpacePyr( CvSize size, int nscales, int minSize, int maxLevels );

/** Compute all levels of scale space */
void cvComputeScaleSpacePyr( CvArr * image, CvScaleSpacePyr * scalespace, float sigma=1.6 );

/** Detect features in an image using difference of gaussian detector */
CvSeq * cvFeaturesDoG( CvScaleSpacePyr * scalespace, CvMemStorage * storage, float dogThresh=3.0 );

/** Refine features.  Orients features, eliminates edge like features, 
 * features with poor contrast and performs sub pixel localization. */
void cvRefineFeatures( CvScaleSpacePyr * scalespace, CvSeq * features, float edgeRatio=5.5, float contrastThresh=0.03);

/** Compute SIFT descriptors for scale space features */
CvSeq * cvComputeSIFT( CvScaleSpacePyr * scalespace, CvSeq * features, CvMemStorage storage );

CvSIFTDescriptor * cvGetSeqElemSIFT( CvSeq * seq, int idx );


template <int size=4, int nbins=8>
class Descriptor : public cv::Array<float, size*size*nbins> {
public:
	float & operator () (const int & i,const int & j, const int & k){
		return (*this)[i*size*nbins+j*nbins+k];
	}
	const float & operator () (const int & i,const int & j, const int & k) const{
		return (*this)[i*size*nbins+j*nbins+k];
	}
	int length(){
		return size;
	}
	Descriptor()
		: cv::Array<float, size*size*nbins>()
		{}
	Descriptor(const cv::Array<float,size*size*nbins> & x)
		: cv::Array<float,size*size*nbins>(x)
		{}
	Descriptor & operator = (const float & x){
		cv::Array<float,size*size*nbins>::operator=(x);
		return (*this);
	}
	double dist(const Descriptor & s) const{
		Descriptor x = s-(*this);
		x = x*x;
		double sum = x.sum();
		return sqrt(sum);
	}
bool operator != (const Descriptor & d) const {
		return !((*this)==d);
	}
	bool operator == (const Descriptor & d) const {
		for(int i=0; i<size*size*nbins; i++){
			if((*this)[i]!=d[i]) return false;
		}
		return true;
	}

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

#endif //__SIFT_DESCRIPTOR_HH
