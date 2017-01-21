#ifndef __SIFT_DESCRIPTOR_HH
#define __SIFT_DESCRIPTOR_HH

#include <cxcore.h>

#include "scale_point.hh"
#include "dog_scale_space.hh"

namespace sift {

template <int size=4, int nbins=8>
class Descriptor : public CvMat {
public:
	float & operator () (const int & i,const int & j, const int & k){
		return this->data.fl[i*size*nbins+j*nbins+k];
	}
	const float & operator () (const int & i,const int & j, const int & k) const{
		return this->data.fl[i*size*nbins+j*nbins+k];
	}
	int length(){
		return size;
	}
	Descriptor(){
		cvInitMatHeader(this, size*size*nbins, 1, CV_8U);
		cvCreateData( this );
	}
	Descriptor( const Descriptor & x ){
		cvInitMatHeader(this, size*size*nbins, 1, CV_8U);
		cvCreateData( this );
		cvCopy(&x, this);
	}
	Descriptor & operator = (const float & x){
		cvSet(this, cvScalar(x));
		return (*this);
	}
	double dist(const Descriptor & s) const{
		return cvNorm(this, &s, CV_L1);
	}
	void normalize(){
		cvScale(this, this, 1/cvNorm(this), 0);
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
	
#define GET3D(mat,s1,s2,s3,i,j,k) ( mat[(i)*(size)*(nbins)+(j)*(nbins)+(k)] )

	// Ix and Iy are pre rotated to orientation of point
	// feature is assumed to be located at the center of image patch
	template <typename image_t>
	void calcFromPatch(const ScalePoint &p, const image_t & Ix, const image_t & Iy) {
		float * data = new float[size*size*nbins];
		CvMat data_mat = cvMat(size*size*nbins,1,CV_32F,data);
		int n = this->length();
		int radius = n*size;
		int xmin = 0;//MAX(0, this->x-radius) - this->x + Ix.width/2;
		int ymin = 0;//MAX(0, this->y-radius) - this->y + Ix.height/2;
		int xmax = Ix.width;//MIN(width-1, this->x+radius) - this->x + Ix.width/2;
		int ymax = Ix.height;//MIN(height-1, this->y+radius) - this->y + Ix.height/2;
		
		cvZero( &data_mat );

		for(int j=ymin; j<ymax; j++){
			int ybin = (j*size)/Ix.height;
			for(int i=xmin; i<xmax; i++){
				int xbin = (i*size)/Ix.width;
				
				float magnitude;
				float angle;
				ScalePoint::calcGradient(i,j,Ix,Iy,magnitude,angle);
				int rbin = ScalePoint::binGradient(angle,nbins);
				GET3D( data, size, size, nbins, xbin, ybin, rbin ) += magnitude;
			}
		}
		// what to scale by ?
		cvScale( &data_mat, this, 255*cvNorm(&data_mat)); 
		delete data;
	}
	

	void calc(ScalePoint &p, const cv::DoGPyramid<float> & dog) {
		calc(p, dog.getIx(p.level, p.subLevel), dog.getIy(p.level, p.subLevel));
	}
	
	// Ix and Iy are standard sobel images from the pyramid, not preprocessed
	template <typename image_t>
	void calc(ScalePoint &p, const image_t & Ix, const image_t & Iy){
		float * data = new float[size*size*nbins];
		CvMat data_mat = cvMat(size*size*nbins,1,CV_32F,data);
		int n = this->length();
		int radius = n*size;
		int xmin = (int) (MAX(0, cvFloor(p.x)-radius));
		int ymin = (int) (MAX(0, cvFloor(p.y)-radius));
		int xmax = (int) (MIN(Ix.width, cvFloor(p.x)+radius));
		int ymax = (int) (MIN(Ix.height, cvFloor(p.y)+radius));
		int xoff = (int) (p.x-radius);
		int yoff = (int) (p.y-radius);
		cvZero( &data_mat );
		for(int j=ymin; j<ymax; j++){
			int ybin = ((j-yoff)*size)/(radius*2);
			for(int i=xmin; i<xmax; i++){
				int xbin = ((i-xoff)*size)/(radius*2);

				float magnitude;
				float angle;
				ScalePoint::calcGradient(i,j,Ix,Iy,magnitude,angle);
				// need to account for angle relative to orientation
				angle-=p.angle;
				int rbin = ScalePoint::binGradient(angle,nbins);
				GET3D( data, size, size, nbins, xbin, ybin, rbin ) += magnitude;

				assert(xbin>=0 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(xbin<4 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(ybin>=0 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
				assert(ybin<4 || fprintf(stderr, "xmin=%d ymin=%d xmax=%d ymax=%d xoff=%d yoff=%d p.x=%f p.y=%f radius=%d\n", xmin, ymin,xmax,ymax,xoff,yoff,p.x,p.y,radius)==0);
			}
		}
		cvScale( &data_mat, this, 255*cvNorm(&data_mat)); 
		delete data;
	}
};

} // namespace sift

#endif //__SIFT_DESCRIPTOR_HH
