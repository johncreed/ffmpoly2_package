CXXFLAGS=-std=c++11 -Wall -fopenmp -O3 -march=native

all: train

train: train.cpp afm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

afm.o: afm.cpp afm.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f train predict afm.o
