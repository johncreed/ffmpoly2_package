CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

DFLAG = -DUSESSE

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: poly2-train

poly2-train: ffm-train.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^

poly2_w-predict: ffm-predict.cpp ffm.o timer.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f poly2-train poly2-predict ffm.o
