#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include <iostream>
#include "cvsift.h"

int main(int argc, char ** argv){
	CvMat * im3;
	CvMat * im;
	CvMat * imf;
	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq * features = NULL;
	cvNamedWindow("p", 1);
	FILE * f = fopen("keys.txt", "w");
	assert(f);

	for(int i=1; i<argc; i++){
		std::cout<<"Opening "<<argv[i]<<std::endl;
		im = cvLoadImageM( argv[i], CV_LOAD_IMAGE_GRAYSCALE);
		im3 = cvLoadImageM( argv[i], CV_LOAD_IMAGE_COLOR);
		std::cout<<"Convert to float: "<<argv[i]<<std::endl;	
		imf = cvCreateMat(im->rows, im->cols, CV_32F);
		cvConvert( im, imf );

		double min, max;
		cvMinMaxLoc(imf, &min, &max);
		printf("%f %f\n", min, max);
		std::cout<<"Detecting features: "<<argv[i]<<std::endl;
		
		// TODO change cvFindFeatures to return sequence since it does not modify it
		features = cvFindFeaturesDoG( imf, storage, 0.0005 ); 
		printf("Found; %d keypoints\n", features->total);
		fprintf(f, "%d 128\n", features->total);
		for(int i=0; i<features->total; i++){
			CvScalePoint * sp = CV_GET_SEQ_ELEM( CvScalePoint, features, i );
			CvPoint2D32f p = cvPointFromScale32f( *sp );

			cvCircle(im3, cvPointFrom32f(p), 2*exp2(sp->level), CV_RGB(0,255,0) );
			fprintf(f, "%f %f %f %f\n", p.x, p.y, 0, sp->angle);
			for(int i=0; i<128; i++){
				fprintf(f, "%d ", 0);
			}
			fprintf(f, "\n");
		}
		cvClearSeq(features );

		cvShowImage("p", im3);
		cvWaitKey(-1);
	}
}
