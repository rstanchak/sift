#ifndef __CV_DOG_SCALE_SPACE_HH
#define __CV_DOG_SCALE_SPACE_HH

#include <vector>
#include <cv/image_pyramid.hh>
#include <cv/scale_space.hpp>

namespace cv {

template <typename pixel_t> class DoGScaleSpace {
	ScaleSpace<pixel_t, 1> _gaussian;
	ScaleSpace<pixel_t, 1> _dog;
	ScaleSpace<pixel_t, 1> _Ix;
	ScaleSpace<pixel_t, 1> _Iy;
	float _freqFactor;
	int _numLevels;
	float _sigma;

public:
	DoGScaleSpace(){
	}
	DoGScaleSpace(int levels):
		_gaussian(levels+3),      // need +1 for difference + 2 for searching
		_dog(levels+2),           // need 2 extra scales for searching
		_Ix(levels+2),
		_Iy(levels+2),
		_freqFactor(stok(levels)),
		_numLevels(levels),
		_sigma(1.6)
		// the octaves are separate by a factor of 2*sigma
		// thus for k levels in each octave,
		// the freqFactor between them is 2^1/k
		// note that to search all scales, we need overlap
		// between scale space octaves.

		// Normally, an octave would have scales sigma -> 2^(k-1)/k*sigma 
		// However, in this case, not all scales are searched.
		//
		// scale       gauss  dog   search
		// s            Y      N       N
		// s*2^1/k      Y      Y       N
		// s*2^2/k      Y      Y       Y
		// ...          Y      Y       Y
		// s*2^(k-1)/k  Y      Y       N
		
		// Specifically, the first two scales don't get searched, and
		// the last doesn't.
		// We make up for this by adding some overlap between octaves

		// ...
		// s*2^(k-1)/k  Y      Y       Y
		// s*2          Y      Y       Y   // also in octave 1
		// s*2*2^1/k    Y      Y       Y   // also in octave 1
		// s*2*2^1/k    Y      Y           // also in octave 1

		// Now s*2^n and s*2^(n+1/k) are searched (except for 2^0 and 2^1/k)
		// and s*2^(n+(k-1)/k) is searched (except for the very end of the pyramid)
		
	{
	}
	DoGScaleSpace(const DoGScaleSpace & s):
		_gaussian(s._gaussian.size()),
		_dog(s._dog.size()),
		_Ix(s._Ix.size()),
		_Iy(s._Iy.size()),
		_freqFactor(s._freqFactor),
		_numLevels(s._numLevels),
		_sigma(s._sigma)
		{
	
	}
	void init(int levels, float freqFactor=-1){
		_gaussian.resize(levels+3); // 2 for search + 1 for difference
		_dog.resize(levels+2);      // need 2 extra scales for searching
		_Ix.resize(levels+2);
		_Iy.resize(levels+2);
		_freqFactor = stok(levels);
		_numLevels = levels;
		_sigma = 1.6;
		std::cout<<_gaussian.size()<<std::endl;
	}

	bool realloc(int w, int h){
		bool ret = _gaussian.realloc(w,h);
		ret = _dog.realloc(w,h) || ret;
		ret = _Ix.realloc(w,h) || ret;
		ret = _Iy.realloc(w,h) || ret;
		return ret;
	}
	static int calcGaussianWidth(float sigma){
		return (((int)(sigma*3))*2 + 1);
	}
	static double stok (int s)
	{
		return pow (2.0, 1.0 / s);
	}

	static void dscale2(const DoGScaleSpace & src, DoGScaleSpace &dest){
		//float sigma=1.6;
		//int width = calcGaussianWidth(sigma);

		//I'll take the LAST image in src, downsample it
		//Image<pixel_t>::dscale2(src._gaussian[src._gaussian.size()-1], dest._gaussian[0]);
		ScaleSpace<pixel_t, 1>::dscale2( src._gaussian, dest._gaussian );
		//ScaleSpace<pixel_t, 1>::dscale2(src._gaussian[0], dest._gaussian[0]);

		//now compute the blurs
		dest = dest._gaussian[0];
	}
	float calcSigma(double k){
		return _sigma * exp2(k/(double)_numLevels);
	}
	const DoGScaleSpace & operator=(const Image<pixel_t> & im){
		int width;
		float sigma;
		assert(_gaussian.size()>0);
		assert(_dog.size()==_gaussian.size()-1);

		// calculate gaussians
		for(size_t i=1; i<_gaussian.size(); i++){
			sigma = calcSigma(i);
			width = calcGaussianWidth(sigma); // 19);
			//width = nwidth-width + 1;
			//width = calcGaussianWidth(_freqFactor); // 19);
			std::cerr<<"Sigma = "<<sigma<<std::endl;
			//std::cerr<<"Convolving with gaussian size "<<width<<std::endl;
			//cvSmooth(&(_gaussian[i-1]), &(_gaussian[i]), CV_GAUSSIAN, width, width);
			cvSmooth(&im, &(_gaussian[i]), CV_GAUSSIAN, 0, 0, sigma);
			
		}
		// first one
		if(&im != &(_gaussian[0])){
			cvSmooth(&im, &(_gaussian[0]), CV_GAUSSIAN, 0, 0, _sigma);
		}
		else{
			//std::cerr<<_gaussian[0].width<<"x"<<_gaussian[0].height<<std::endl;
			cvSmooth(&(_gaussian[0]), &(_gaussian[0]), CV_GAUSSIAN, 0, 0, _sigma);
		}

		// now dog's and gradients
		for(size_t i=1; i<_gaussian.size(); i++){		
			_dog[i-1] = _gaussian[i] - _gaussian[i-1];
			cvSobel(&_gaussian[i-1], &_Ix[i-1], 1, 0);
			cvSobel(&_gaussian[i-1], &_Iy[i-1], 0, 1);
		}
		
		return (*this);
	}

	Image<pixel_t> & operator[] (const int & i){
		return _dog[i];
	}

	const Image<pixel_t> & operator[] (const int & i) const{
		return _dog[i];
	}
	const Image<pixel_t> & getGaussian(const int & i) const{
		return _gaussian[i];
	}
	const Image<pixel_t> & getIx(const int & i) const{
		return _Ix[i];
	}
	const Image<pixel_t> & getIy(const int & i) const{
		return _Iy[i];
	}
	
	int dog_size() const { return _dog.size(); }
	int size() const { return _gaussian.size(); }
	const int &getWidth() const { return _dog[0].width; }
	const int &getHeight() const { return _dog[0].height; }

};

template <typename pixel_t> class DoGPyramid : public ImagePyramid< DoGScaleSpace<pixel_t> > {
	typedef Image<pixel_t> image_t;
public:
	DoGPyramid(int minImageSize=16, int nSubScales=1):
		ImagePyramid< DoGScaleSpace<pixel_t> > (minImageSize, DoGScaleSpace<pixel_t>(nSubScales))
	{
	}
	const image_t & getImage(int i, int j) const{
		return (*this)[i][j];
	}
	image_t & getImage(int i, int j){
		return (*this)[i][j];
	}
	const image_t & getGaussian(int i, int j) const{
		return (*this)[i].getGaussian(j);
	}
	const image_t & getIx(const int &i, const int &j) const {
		assert(i<this->getNumScales());
		assert(j<this->getNumSubScales());
		return (*this)[i].getIx(j-1);
	}
	const image_t & getIy(const int &i, const int &j) const {
		assert(i<this->getNumScales());
		assert(j<this->getNumSubScales());
		return (*this)[i].getIy(j-1);
	}
	int getNumScales() const { return ImagePyramid<DoGScaleSpace<pixel_t> >::_images.size(); }
	int getNumSubScales() const { return ImagePyramid<DoGScaleSpace<pixel_t> >::_images[0].dog_size(); }
	/** calculate effective sigma for this subspace */
	float getSigma(int i, int j){
		return pow(2.0, (double)i)*ImagePyramid<DoGScaleSpace<pixel_t> >::_images[i].calcSigma(j);
	}
};

} //namespace cv
#endif //__CV_DOG_SCALE_SPACE_HH
