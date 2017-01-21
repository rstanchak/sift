#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <iostream>
#include "cvsift.h"

int main(int argc, char ** argv){
	
	CvMat * im8, *im;
	CvScaleSpacePyr * space, *dogspace;
	cvNamedWindow("win",0);
	cvNamedWindow("dog",0);
	CvMat * grey = cvCreateMat(512,512, CV_32F);
	for(int k=1; k<argc; k++){
		assert( im = cvLoadImageM(argv[1], CV_LOAD_IMAGE_GRAYSCALE ) );
		CvSize sz = cvGetSize( im );
		sz.width=sz.width*2-2;
		sz.height=sz.height*2-2;
		space = cvCreateScaleSpacePyr( sz, 6, 16, 5 );
		dogspace = cvCreateScaleSpacePyr( sz, 5, 16, 5 );
		cvConvert( im, grey );
		cvComputeScaleSpacePyr( grey, space, 1.6  );
		cvComputeDoGScaleSpacePyr( space, dogspace );
		for(int i=0; i<space->nLevels; i++){
			for( int j=0; j<space->levels[i].nScales-1; j++){
				printf("%d %d sigma=%f\n", i, j, cvScaleSpaceSigma(space, i, j) );
				CvMat * mat = space->levels[i].images[j];
				cvResize(mat, grey);
				cvNormalize(grey, grey, 0, 1, CV_MINMAX);
				cvShowImage("win", grey);
				mat = dogspace->levels[i].images[j];
				cvResize(mat, grey);
				cvNormalize(grey, grey, 0, 1, CV_MINMAX);
				cvShowImage("dog", grey);
				cvWaitKey(-1);
			}
		}
	}
}
