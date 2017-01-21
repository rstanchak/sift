#ifndef __DOG_FEATURE_DETECTOR_HPP
#define __DOG_FEATURE_DETECTOR_HPP

#include <cv/fixed_priority_queue.hpp>
#include <list>
#include <cv/image_pyramid.hh>
#include <cv/image.h>
#include <cv/cvext.hpp>

// standard dscale2 uses nearest neighbor -- want smoothing
template <typename T, int nch=1>
class PyrImage : public cv::Image<T, nch> {
public:
	PyrImage<T,nch> & operator= ( const cv::Image<T, nch> & im ){
		cv::Image<T,nch>::operator=( im );
		return (*this);
	}
	cv::Image<T,nch> & operator() () {
		return * (cv::Image<T,nch>) this;
	}
	PyrImage<T,nch> & operator= ( const PyrImage<T,nch> & im ){
		cv::Image<T,nch>::operator=( * (cv::Image<T,nch> *)this );
		return (*this);
	}
	/** downsampes image by a factor of 2 */
	static cv::Image<T, nch> & dscale2(const cv::Image<T,nch> & a, cv::Image<T,nch> & b) {
		int w=a.width,h=a.height;
		b.realloc(w/2,h/2);
		cvPyrDown(&a, &b);
		return b;
	}	
};


template <typename data_t>
class DoGFeatureDetector : public std::list<CvCircle32f> {
protected:
	cv::ImagePyramid< PyrImage< data_t> > pyrdown;  // this pyramid uses smoothed dscale above
	cv::ImagePyramid< cv::Image< data_t > > pyrup;
	cv::ImagePyramid< cv::Image< uchar, 1 > > pyrismax;
	bool m_invert; // true indicates find dark on white .. false white on dark
	int m_maxPoints;
public:
	DoGFeatureDetector( int size=5 ) : pyrdown(4), pyrup(4), pyrismax(4), m_invert(false), m_maxPoints( size )
	{
	}
	void push_points( cv::Image<data_t> & dImage, cv::Image<uchar> & isMax, std::list<CvCircle32f> & points,
			int level ){

		fixed_priority_queue< CvPoint > queue( m_maxPoints );
		

		for(int j=0; j<dImage.height; j++){
			for(int i=0; i<dImage.width; i++){
				if(isMax[j][i]>0){
					queue.push( cvPoint( i, j ), dImage[j][i]); 
				}
			}
		}

		//std::cout<<"queue size: "<<queue.size()<<std::endl;
		while( !queue.empty() ){
		//	printf("%d %d %f\n", n.p.x, n.p.y, n.val);
			CvPoint p = queue.top().data;
			points.push_back( cvCircle32f( (p.x+0.49) * (1<<level), (p.y+0.49) * (1<<level), (2<<level)));
			queue.pop();
		}
	}
	void setInvert(bool inv){
		m_invert = inv;
	}

	void detect( cv::Image<data_t> & src ){
		this->clear();
		pyrdown.calculate( src );
		pyrup.checkSizes( src );
		pyrismax.checkSizes( src );
		for (int k=1; k<pyrdown.size(); k++){
			pyrdown[k].resize(pyrup[k-1]);
			//pyrup[k-1].imagesc("ir");
			//cvWaitKey(-1);

			// could move this loop into the push loop
			if(m_invert){
				pyrup[k-1] = pyrup[k-1] - pyrdown[k-1];
			}
			else{
				pyrup[k-1] = pyrdown[k-1] - pyrup[k-1];
			}
			cvIsLocalMax( pyrup[k-1], pyrismax[k-1] );
			this->push_points( pyrup[k-1], pyrismax[k-1], *this, k-1); 
		}

		//std::cout<<"points size: "<<this->size()<<std::endl;
		//printf("Image size: %dx%d\n", src.width, src.height);
	}
	void draw( IplImage * im, const CvScalar & color ){
		//std::cout<<"Items #="<<this->size()<<std::endl;
		for( iterator it = this->begin(); it!=this->end(); ++it){
			cvCircle( im, cvPointFrom32f(it->c), (int)it->r, color, 1, CV_AA );
		}
	}
	cv::Image< data_t > & diffImage( int level ) {
		return pyrup[level];
	}
};

#endif //__DOG_FEATURE_DETECTOR_HPP
