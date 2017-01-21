CXXFLAGS=-ggdb `pkg-config opencv --cflags`
LDFLAGS=`pkg-config opencv --libs`
LINK=$(CC) -o $@ $(LDFLAGS) $^
dog_scale_space: dog_scale_space.o cvsift.o 
	$(LINK)
sift_feature_track: sift_feature_track.o 
	$(LINK)
point_correspond_sift: point_correspond_sift.o 
	$(LINK)
sift_feature_detector: sift_feature_detector.o 
	$(LINK)
dog_feature_detector2: dog_feature_detector2.o 
	$(LINK)
dog_feature_detector: dog_feature_detector.o cvsift.o
	$(LINK)
scale_view: scale_view.o 
	$(LINK)
clean:
	rm -f *.o  


