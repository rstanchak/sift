#ifndef CV_SIFT_H 
#define CV_SIFT_H
#include <cxtypes.h>

struct CvScalePoint {
	CvPoint2D32f center; // in world coordinates
	int level;
	int subLevel;
	float angle;
};

CV_INLINE CvScalePoint cvScalePoint( float x, float y, int level=0, int subLevel=0, float angle=0 ){
	CvScalePoint sp;
	sp.center.x = x; 
	sp.center.y = y;
	sp.angle = angle;
	sp.level = level;
	sp.subLevel=subLevel;
	return sp;
}

struct CvScaleSpace {
	CvMat ** images;
	float sigma;
	int nScales;
};

struct CvScaleSpacePyr {
	CvScaleSpace * levels;
	float sigma;
	int nLevels;
};

CV_INLINE CvPoint2D32f cvPointFromScale32f( CvScalePoint p ){
	CvPoint2D32f q;
	q.x = exp2( p.level-1 ) * p.center.x;
	q.y = exp2( p.level-1 ) * p.center.y;
	return q;
}
CV_INLINE CvPoint cvPointFromScale( CvScalePoint p ){
	CvPoint q;
	q.x = cvFloor( exp2( p.level-1 ) * p.center.x );
	q.y = cvFloor( exp2( p.level-1 ) * p.center.y );
	return q;
}

struct CvSIFTDescriptor {
	CvMat hist;  // of size rbins*sbins*sbins
	CvScalePoint point;
	int rbins; //# of radial bins
	int sbins; //# of spatial bins
};

CvSIFTDescriptor * cvCreateSIFTDescriptor(int sbins, int rbins);

/** Allocate scale space data structure */
CvScaleSpacePyr * cvCreateScaleSpacePyr( CvSize size, int nscales, int minSize, int maxLevels );
void cvReleaseScaleSpacePyr( CvScaleSpacePyr ** scalespace_ptr );

/** Compute all levels of scale space */
void cvComputeScaleSpacePyr( const CvArr * image, CvScaleSpacePyr * scalespace, float sigma=1.6 );


CV_INLINE float cvScaleSpaceSigma( CvScaleSpacePyr * scalespace, int level, int sublevel ){
	return scalespace->sigma*(level) + 
		    scalespace->sigma*(sublevel-1)/(float)(scalespace->levels[level].nScales-3);
}

void cvComputeDoGScaleSpacePyr( CvScaleSpacePyr * scalespace, CvScaleSpacePyr * dogspace );

/** Detect features in an image using difference of gaussian detector */
CvSeq * cvFindFeaturesDoG(  const CvArr * im, CvMemStorage * storage, 
		float dogThresh=0.0, float edgeRatio=20.0, float contrastThresh=0.03, float sigma=1.6,    
		int minImageSize=24, int nsub=3 );

/** Refine features.  Orients features, eliminates edge like features, 
 * features with poor contrast and performs sub pixel localization. */
void cvRefineFeatures( CvScaleSpacePyr * scalespace, CvSeq * features, float edgeRatio=5.5, float contrastThresh=0.03);

/** Compute SIFT descriptors for scale space features */
CvSeq * cvComputeSIFT( CvScaleSpacePyr * scalespace, CvSeq * features, CvMemStorage storage );

CvSIFTDescriptor * cvGetSeqElemSIFT( CvSeq * seq, int idx );

#endif // CV_SIFT_H
