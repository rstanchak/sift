#ifndef __DOG_FEATURE_DETECTOR_HH
#define __DOG_FEATURE_DETECTOR_HH

#include <list>
#include <cv/wincompat.h>
#include <cv/dog_scale_space.hh>
#include <cv/array.h>
#include <cv/matrix.hh>
#include <cv/scale_point.hh>
#include "cv.h"


typedef cv::Image<float> image_t;



/// Difference - of - Gaussians (DoG) based feature detector
class DoGFeatureDetector{
protected:
	
	cv::DoGPyramid<float> _spaces;
	std::list < ScalePoint > _features;
	float _dogThresh;
	float _edgeRatio;
	float _contrastThresh;
	float _sigma;
	bool _peaksOnly;

public:
	typedef ScalePoint feature_t;
	typedef std::list<feature_t>::iterator const_iterator;
	typedef std::list<feature_t>::iterator iterator;

	/** constructor
	*/
	DoGFeatureDetector(float dogThresh=3.0, float edgeRatio=5.5, float contrastThresh=0.03, float sigma=1.6, int minImageSize=16, int nsub=1):
		_spaces(minImageSize, nsub),
		_dogThresh(dogThresh), // was 10.0
		_edgeRatio(edgeRatio),     // was 5.5
		_contrastThresh(contrastThresh),
		_sigma(sigma),
		_peaksOnly(true)
	{
		//std::cout<<"DoGFeatureDetector:"<<std::endl;
		//std::cout<<"\tdogThresh="<<_dogThresh<<std::endl;
		//std::cout<<"\tedgeRatio="<<_edgeRatio<<std::endl;
		//std::cout<<"\tsigma="<<_sigma<<std::endl;
		//std::cout<<"\tminImageSize="<<minImageSize<<std::endl;
		//std::cout<<"\tnsub="<<nsub<<std::endl;
	}
	void setDoGThreshold(float t){
		_dogThresh=t;
	}
	void setEdgeRatio(float r){
		_edgeRatio=r;
	}
	void setContrastThreshold(float t){
		_contrastThresh=t;
	}
	const cv::Image<float> & getGaussian(const int & i, const int & j) const{
		return _spaces.getGaussian(i,j);
	}
	
	cv::DoGPyramid<float> & getPyramid(){
		return _spaces;
	}
	
	/** find and track features */
	void track(const cv::Image<float> & im){
		// if I have no saved features, detect some
		if(_features.size()<=0){
			this->detect(im);
			return;
		}
		
		// update scale space representation
		this->update(im);

		// now update locations of features using subpixel localize
		for(iterator it = _features.begin(); it!=_features.end(); ++it){
			this->subPixelLocalize(*it);	
		}	
	}
	
	/** update scale space
	*/
	void update(const cv::Image<float> & im){
		cv::Image<float> f;
		cv::Image<float>::uscale2(im, f);
		_spaces.calculate(f);
	}
	
	/** find features at one level -- assume update has already been called
	*/
	void detect_one(int level){
		cv::Image<uchar> gt,lt,tmp;
		gt.reshape(_spaces.getImage(level,0).width-2, _spaces.getImage(level,0).height-2);
		lt.reshape(gt.width, gt.height);
		tmp.reshape(gt.width, gt.height);
		for(int j=1; j<_spaces.getNumSubScales()-1; j++){
			findPeaks3LevelFuzzy(_spaces.getImage(level,j-1),
					_spaces.getImage(level,j),
					_spaces.getImage(level,j+1),
					gt, lt, tmp,
					_dogThresh, level, j, 15);
		}
	}
	void clear(){
		_features.clear();
	}
	void detect(const cv::Image<float> &im, int level=-1){
		
		this->update(im);
		
		// clear old features
		_features.clear();
		
		
		if(level==-1){
			
			// find peaks at each scale
			//std::cout<<"Searching for peaks"<<std::endl;
			for(int i=_spaces.getNumScales()-1; i>=0; i--){
				this->detect_one(i);
			}
		}
		/*else{
			// update DoG pyramid
			_spaces.calculate(im, level);
			
			gt.reshape(_spaces.getImage(level,0).width-2, _spaces.getImage(level,0).height-2);
			lt.reshape(gt.width, gt.height);
			tmp.reshape(gt.width, gt.height);
			for(int j=1; j<_spaces.getNumSubScales()-1; j++){
				findPeaks3Level(_spaces.getImage(level,j-1),
						_spaces.getImage(level,j),
						_spaces.getImage(level,j+1),
						gt, lt, tmp,
						_dogThresh, level, j);
			}
		}
*/
	}
	void filter() {
		int i=0;
		
		// gradient patches for orientation assignment
		image_t Ix(16,16),Iy(16,16);
		
		// transformation matrix for extracting image patches
		fMat A(2,3);
		
		// filter out bad points
		for(iterator it = _features.begin(); 
		    it!=_features.end();
			){
			//std::cout<<"Filtering feature: "<<*it<<std::endl;
			cv::Image<float> im = _spaces.getImage(it->level, it->subLevel);
			it->x+=.49;
			it->y+=.49;
			// Lowe uses an r value of 10
			// Here we are comparing (r+1)^2/r to the ratio of largest direction of curvature 
			// to smallest curvature, so larger values let in more edges
			//std::cout<<"Feature "<<i<<" - "<<*it<<" - ";
			if(filterEdge(_spaces.getImage(it->level, it->subLevel), it->x, it->y, _edgeRatio)){
				iterator it2 = it;
				it++;
				_features.erase(it2);
				//std::cout<<"Rejected -- too edge-like"<<std::endl;
			}
			// this step ought to spawn other features if the orientation is ambiguous
			else if(!subPixelLocalize(*it)){
				iterator it2 = it;
				it++;
				_features.erase(it2);
				//std::cout<<"Rejected -- can't localize"<<std::endl;
			}
			else if(filterContrast(*it, _contrastThresh)){
				iterator it2 = it;
				it++;
				_features.erase(it2);
				//std::cout<<"Rejected -- poor contrast"<<std::endl;
			}
			else{
				A(0,0)=1; A(0,1)=0; A(0,2)=it->x;
				A(1,0)=0; A(1,1)=1; A(1,2)=it->y;
				insert_iterator< std::list<ScalePoint> > ii(_features, it);
				iterator it2 = it;
				++it;
				cvGetQuadrangleSubPix(&_spaces.getIx(it->level,it->subLevel), &Ix, &A);
				cvGetQuadrangleSubPix(&_spaces.getIy(it->level,it->subLevel), &Iy, &A);
				it2->calcOrientations(Ix,Iy, _spaces.getIx(it2->level,it2->subLevel).width, _spaces.getIx(it2->level, it2->subLevel).height, ii);
				//std::cout<<"Good feature"<<std::endl;
			}
			i++;
		}	
	}

	/*const inline feature_t & operator [] (const int & i) const {
		return _features[i];
	}
	*/
	int size() const { return _features.size(); }
	
	iterator begin() { return _features.begin(); }
	iterator end() { return _features.end(); }
	
	// hack to compare one section of an image to that of another
	void compare(IplImage * cur, IplImage * im, int x, int y,
			cv::Image<uchar> &gt, cv::Image<uchar> &lt, cv::Image<uchar> &tmp){
		bool cen = (x==1 && y==1);
		//std::cout<<"compare("<<x<<","<<y<<")"<<std::endl;
		cvSetImageROI(im, cvRect(x,y,tmp.width,tmp.height));
		cvCmp(cur, im, &tmp, cen ? CV_CMP_GE : CV_CMP_GT);
		cvAnd(&gt, &tmp, &gt);
		cvCmp(cur, im, &tmp, cen ? CV_CMP_LE : CV_CMP_LT);
		cvAnd(&lt, &tmp, &lt);
		cvResetImageROI(im);
	}
	
	void compare_fuzzy(IplImage * cur, IplImage * im, int x, int y,
			cv::Image<uchar> &gt, cv::Image<uchar> &lt, cv::Image<uchar> &tmp){
		bool cen = (x==1 && y==1);
		//std::cout<<"compare("<<x<<","<<y<<")"<<std::endl;
		cvSetImageROI(im, cvRect(x,y,tmp.width,tmp.height));
		cvCmp(cur, im, &tmp, cen ? CV_CMP_GE : CV_CMP_GT);
		cvAddS(&gt, cvScalarAll(1), &gt, &tmp);
		cvCmp(cur, im, &tmp, cen ? CV_CMP_LE : CV_CMP_LT);
		cvAddS(&lt, cvScalarAll(1), &lt, &tmp);
		cvResetImageROI(im);
	}
	

	// find the _features in the given level of the DoG space -- need scales above and below
	// to determine 'maximum-ness'
	void findPeaks3LevelFuzzy(const cv::Image<float> & below,
			const cv::Image<float> & current,
			const cv::Image<float> & above,
			cv::Image<uchar> & gt,
			cv::Image<uchar> & lt,
			cv::Image<uchar> & cmp,
			float dogThresh,
			int level, int sublevel, int fuzzyThresh){
		//std::cout<<"findPeaks3Level("<<level<<","<<sublevel<<")"<<std::endl;
		IplImage cur, tmp;
		//std::vector< feature_t > _features;

		//float scale = (float)(exp2(level-1));

		// copy header of current image
		memcpy(&cur, &current, sizeof(IplImage));

		//manipulate ROI of copy to do bulk comparisons
		cvSetImageROI(&cur, cvRect(1,1, current.width-2, current.height-2));

		// 
		double min,max;
		cvMinMaxLoc(&cur, &min, &max);
		std::cerr<<"maxima: "<<min<<","<<max<<std::endl;
		
		// threshold first 
		// cvInRangeS(&cur, cvScalar(-dogThresh), cvScalar(dogThresh), &gt);
		//cvNot(&gt, &gt);
		cvZero(&lt);
		gt = lt;

		// current level
		memcpy(&tmp, &current, sizeof(IplImage));
		compare_fuzzy(&cur, &tmp, 0, 0, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 0, 1, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 0, 2, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 1, 0, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 1, 2, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 2, 0, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 2, 1, gt, lt, cmp);
		compare_fuzzy(&cur, &tmp, 2, 2, gt, lt, cmp);

		// upper level
		memcpy(&tmp, &above, sizeof(IplImage));
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				compare_fuzzy(&cur, &tmp, i, j, gt, lt, cmp);
			}
		}

		// lower level
		memcpy(&tmp, &below, sizeof(IplImage));
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				compare_fuzzy(&cur, &tmp, i, j, gt, lt, cmp);
			}
		}

		int h=gt.height-1,w=gt.width-1;
		unsigned char * g, *l;

		// find maximums & minimums -- ignore boundary pixels
		float sigma = _spaces[level].calcSigma(sublevel);
		float imgScale = exp2(level-1)*sigma;
		for(int j=1; j<h; j++){
			g = gt.prow(j);
			l = lt.prow(j);
			for(int i=1; i<w; i++){
				//std::cout<<(int)l[i]<<std::endl;
				if(l[i]>=fuzzyThresh || g[i]>=fuzzyThresh){
					_features.push_back( feature_t( i+1, j+1, level, sublevel, sigma, imgScale));
					//_features.push_back( feature_t((i+0.5)*scale, (j+0.5)*scale, scale,scale));
				}
			}
		}
		//std::cout<<"Found "<<_features.size()<<" raw _features at level ("<<level<<","<<sublevel<<")"<<std::endl;
		//return _features;
	}
	// find the _features in the given level of the DoG space -- need scales above and below
	// to determine 'maximum-ness'
	void findPeaks3Level(const cv::Image<float> & below,
			const cv::Image<float> & current,
			const cv::Image<float> & above,
			cv::Image<uchar> & gt,
			cv::Image<uchar> & lt,
			cv::Image<uchar> & cmp,
			float dogThresh,
			int level, int sublevel){
		//std::cout<<"findPeaks3Level("<<level<<","<<sublevel<<")"<<std::endl;
		IplImage cur, tmp;
		//std::vector< feature_t > _features;

		//float scale = (float)(exp2(level-1));

		// copy header of current image
		memcpy(&cur, &current, sizeof(IplImage));

		//manipulate ROI of copy to do bulk comparisons
		cvSetImageROI(&cur, cvRect(1,1, current.width-2, current.height-2));

		// 
		double min,max;
		cvMinMaxLoc(&cur, &min, &max);
		std::cerr<<"maxima: "<<min<<","<<max<<std::endl;
		
		// threshold first 
		cvInRangeS(&cur, cvScalar(-dogThresh), cvScalar(dogThresh), &gt);
		cvNot(&gt, &gt);
		lt = gt;

		// current level
		memcpy(&tmp, &current, sizeof(IplImage));
		compare(&cur, &tmp, 0, 0, gt, lt, cmp);
		compare(&cur, &tmp, 0, 1, gt, lt, cmp);
		compare(&cur, &tmp, 0, 2, gt, lt, cmp);
		compare(&cur, &tmp, 1, 0, gt, lt, cmp);
		compare(&cur, &tmp, 1, 2, gt, lt, cmp);
		compare(&cur, &tmp, 2, 0, gt, lt, cmp);
		compare(&cur, &tmp, 2, 1, gt, lt, cmp);
		compare(&cur, &tmp, 2, 2, gt, lt, cmp);

		// upper level
		memcpy(&tmp, &above, sizeof(IplImage));
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				compare(&cur, &tmp, i, j, gt, lt, cmp);
			}
		}

		// lower level
		memcpy(&tmp, &below, sizeof(IplImage));
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				compare(&cur, &tmp, i, j, gt, lt, cmp);
			}
		}

		int h=gt.height-1,w=gt.width-1;
		unsigned char * g, *l;

		// find maximums & minimums -- ignore boundary pixels
		float sigma = _spaces[level].calcSigma(sublevel);
		float imgScale = exp2(level-1)*sigma;
		for(int j=1; j<h; j++){
			g = gt.prow(j);
			l = lt.prow(j);
			for(int i=1; i<w; i++){
				if((l[i]!=0 && !_peaksOnly) || (g[i]!=0 && _peaksOnly)){
					_features.push_back( feature_t( i+1, j+1, level, sublevel, sigma, imgScale));
					//_features.push_back( feature_t((i+0.5)*scale, (j+0.5)*scale, scale,scale));
				}
			}
		}
		//std::cout<<"Found "<<_features.size()<<" raw _features at level ("<<level<<","<<sublevel<<")"<<std::endl;
		//return _features;
	}

	/*void findPeaks(const cv::DoGScaleSpace<float> &dog,  float dogThresh){
		cv::Image<uchar> gt,lt,tmp;
		for(int i=0; i<dog.getNumScales(); i++){
			//std::cout<<"Looking at level "<<i<<std::endl;
			gt.reshape(dog.getImage(i,0).width-2, dog.getImage(i,0).height-2);
			lt.reshape(gt.width, gt.height);
			tmp.reshape(gt.width, gt.height);
			for(int j=1; j<dog.getNumSubScales()-1; j++){
				_features.push_back(std::vector<feature_t>);
				findPeaks3Level(dog.getImage(i,j-1),
						dog.getImage(i,j),
						dog.getImage(i,j+1),
						gt, lt, tmp,
						dogThresh, i, j));
			}
		}
		//std::cout<<"Found "<<_features.size()<<" raw _features"<<std::endl;
		//return _features;
	}*/

	// Return adjustment (scale, y, x) on success,
	fMat calcAdjustment (const cv::Image<float> & below, 
			const cv::Image<float> & current, 
			const cv::Image<float> & above,
			const int &x, 
			const int &y, double & dp){

		/*Console.WriteLine ("GetAdjustment (point, {0}, {1}, {2}, out double dp
		  )",
		  level, x, y);*/
		dp = 0.0;


		// Hessian
		fMat H(3,3);
		H[0][0] = below[y][x] - 2 * current[y][x] + above[y][x];
		H[0][1] = H[1][0] = 0.25 * (above[y+1][x] - above[y-1][x] -
				(below[y + 1][x] - below[y - 1][x]));
		H[0][2] = H[2][0] = 0.25 * (above[y][x + 1] - above[y][x - 1] -
				(below[y][x + 1] - below[y][x - 1]));
		H[1][1] = current[y - 1][x] - 2 * current[y][x] + current[y + 1][x];
		H[1][2] = H[2][1] = 0.25 * (current[y + 1][x+1] - current[y + 1][x - 1] -
				(current[y - 1][x + 1] - current[y - 1][x - 1]));
		H[2][2] = current[y][x - 1] - 2 * current[y][x] + current[y][x + 1];

		// derivative
		fMat d(3,1);
		d[0][0] = 0.5 * (above[y][x] - below[y][x]);             //dS
		d[1][0] = 0.5 * (current[y + 1][x] - current[y - 1][x]); //dy
		d[2][0] = 0.5 * (current[y][x + 1] - current[y][x - 1]); //dx

		// Solve: H*b = -d --> b = inv(H)*d
		fMat b = H.inv()*d;
		b*=-1;

		dp = b.dot(d);

		return b;
	}
	bool subPixelLocalize(feature_t & p, int nadjustments=2){
		bool needToAdjust = true;
		int x = p.x;
		int y = p.y;
		int level = p.level;
		int sublevel = p.subLevel;
		//std::cerr<<"calcSubPixel Start: "<<p<<std::endl;
		while (needToAdjust) {

			// Points we cannot say anything about, as they lie on the border
			// of the scale space 
			if (sublevel <= 0 || sublevel >= (_spaces.getNumSubScales() - 1))
				return false;

			if (x <= 0 || x >= (_spaces.getImage(level,sublevel).getWidth() - 1))
				return false;
			if (y <= 0 || y >= (_spaces.getImage(level,sublevel).getHeight() - 1))
				return false;

			double dp;
			fMat adj = calcAdjustment (_spaces.getImage(level, sublevel-1),
					_spaces.getImage(level, sublevel),
					_spaces.getImage(level, sublevel+1), x, y, dp);

			// Get adjustments and check if we require further adjustments due
			// to pixel level moves. If so, turn the adjustments into real
			// changes and continue the loop. Do not adjust the plane, as we
			// are usually quite low on planes in thie space and could not do
			// further adjustments from the top/bottom planes.
			double adjS = adj[0][0];
			double adjY = adj[1][0];
			double adjX = adj[2][0];
			if (fabs (adjX) > 0.5 || fabs (adjY) > 0.5 || fabs(adjS) > 0.5) {
				// Already adjusted the last time, give up
				if (nadjustments == 0) {
					//Console.WriteLine ("too many adjustments, returning");
					return false;
				}

				nadjustments -= 1;

				// Check that just one pixel step is needed, otherwise discard
				// the point
				double distSq = adjX * adjX + adjY * adjY;
				if (distSq > 2.0)
					return (false);

				x = (int)(x + adjX + 0.5);  
				y = (int)(y + adjY + 0.5); 
				sublevel = (int)(sublevel + adjS + 0.5);

				//point.Level = (int) (point.Level + adjS + 0.5);
			//	std::cerr<<"moved point by ("<<adjX<<","<<adjY<<": "<<adjS<<
				           //") to ("<<x<<","<<y<<": "<<sublevel<<")"<<std::endl;
				/*Console.WriteLine ("moved point by ({0},{1}: {2}) to ({3},{4}:
				  {5})",
				  adjX, adjY, adjS, point.X, point.Y, point.Level);*/
				continue;
			}

			p.x = x + adjX;
			p.y = y + adjY;
			p.subLevel = sublevel + adjS + 0.5;
			p.scale = _spaces[level].calcSigma(sublevel+adjS);
			p.value = _spaces[(int)level][(int)sublevel][(int)p.y][(int)p.x] + 0.5*dp;
			
			p.recalc(); // recalculates imgScale, radius

			//std::cerr<<"calcSubPixel: "<<p<<std::endl;
			
			//p.scale = _spaces[level].getSigma(
			//p.scale = exp2( (level + adjS)/_spaces.getNumSubScales()) * 1.6;
			//FIXME: csharp has p.value = space[point.X, point.Y] + 0.5 * dp;
			/* for procesesing with gnuplot
			 *
			 Console.WriteLine ("{0} {1} # POINT LEVEL {2}", point.X,
			 point.Y, basePixScale);
			 Console.WriteLine ("{0} {1} {2} # ADJ POINT LEVEL {3}",
			 adjS, adjX, adjY, basePixScale);
			 */

			// Check if we already have a keypoint within this octave for this
			// pixel position in order to avoid dupes. (Maybe we can move this
			// check earlier after any adjustment, so we catch dupes earlier).
			// If its not in there, mark it for later searches.
			//
			// FIXME: check why there does not seem to be a dupe at all
			//if (processesed[point.X, point.Y] != 0)
			//	return (true);

			//processesed[point.X, point.Y] = 1;

			// Save final sub-pixel adjustments.
			//PointLocalInformation local = new PointLocalInformation (adjS, adjX,
			//		adjY);
			//local.DValue = dp;
			//local.DValue = space[point.X, point.Y] + 0.5 * dp;
			//point.Local = local;

			//needToAdjust = false;
			return true;
		}

		return false;

	}

	bool filterContrast(const feature_t &p, float thresh){
		//std::cerr<<"filterContrast - "<<p.value<<std::endl;
		return fabs(p.value) <= thresh;
	}

	bool filterEdge(const cv::Image<float> & space, int x, int y, double r){
		double D_xx, D_yy, D_xy;

		// first check bounds
		if(x<=0 || y<=0 || x>=space.width-1 || y>=space.height-1){
			return false;
		}
		
		// Calculate the He_ssian H elements [ D_xx, D_xy ; D_xy , D_yy ]
		D_xx = space[y][x+1] + space[y][x-1] - 2.0 * space[y][x];
		D_yy = space[y + 1][x] + space[y - 1][x] - 2.0 * space[y][x];
		D_xy = 0.25 * ((space[y + 1][x + 1] - space[y - 1][x + 1]) -
				(space[y + 1][x - 1] - space[y - 1][x - 1]));

		// page 13 in Lowe's paper
		double TrHsq = D_xx + D_yy;
		TrHsq *= TrHsq;
		double DetH = D_xx * D_yy - (D_xy * D_xy);

		double r1sq = (r + 1.0);
		r1sq *= r1sq;

		//fprintf(stderr, "x=%d, y=%d, TrHsq * r = %f, DetH * r1sq = %f, TrHsq / DetH = %f, r1sq /r = %f\n",
		//		x, y, (TrHsq * r), (DetH * r1sq),
		//		(TrHsq / DetH) , (r1sq / r));
		
		// reject if ratio of curvatures is greater than threshold
		if ( fabs(TrHsq / DetH) > (r1sq / r)) {
			return true;
		}

		return false;
	}
	void orientFeature(feature_t &p){
		int nbins = 36;
		int hist[36];
		double sigma = p.scale * 3.0;
		double factor = 2.0*sigma*sigma;
		int radius = (int)(3.0*sigma / 2.0 + 0.5);
		int radiusSq = radius*radius;
		
		bzero(hist, sizeof(int)*36);
		
		int w = _spaces.getGaussian(p.level,p.subLevel).getWidth();
		int h = _spaces.getGaussian(p.level,p.subLevel).getWidth();
		int xmin = MAX(1, p.x-radius);
		int ymin = MAX(1, p.y-radius);
		int xmax = MIN(w-1, p.x+radius);
		int ymax = MIN(h-1, p.y+radius);

		for(int i=xmin; i<xmax; i++){
			for(int j=ymin; j<ymax; j++){
				int relX = i - p.x;
				int relY = j - p.y;

				// only consider points in circle
				double d = relX*relX + relY*relY;
				if(d>radiusSq) continue;
				
				// gaussian weight
				double gWeight = exp(  -( d / factor)  );
				
				int bin;
				float weight;
				binFeature(i,j,_spaces.getGaussian(p.level,p.subLevel), nbins, weight, bin);
				hist[bin] += weight*gWeight;
			}
		}
		
		// find maximum bin
		double maxv=hist[0];
		int maxb=0;
		for(int b=1; b<nbins; b++){
			if(hist[b]>maxv){
				maxv=hist[b];
				maxb=b;
			}
		}
		p.angle = (M_PI*2*maxb)/nbins;
	}
	void binFeature(const int &x, const int &y, const cv::Image<float> & space, const int &nbins, 
	                float & weight, int & bin){
		float angle;
		calcGradient(x,y, space, weight, angle);
		while(angle<0) angle+=(2*M_PI);
		bin = nbins*angle*.5/M_PI;
	}

	void calcGradient(const int &x, const int &y, const cv::Image<float> & space, 
	                  float & magnitude, float & angle){
		// compute gradient magnitude, orientation
		float dx = space[y][x+1]-space[y][x-1];
		float dy = space[y+1][x]-space[y-1][x];
		magnitude = sqrt( dx*dx + dy*dy );
		angle = atan2(dy,dx);
	}
};
#endif //__DOG_DETECTOR_HH

