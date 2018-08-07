CXX = g++
CXXFLAGS = -Wall -O3 -std=c++0x -march=native

# comment the following flags if you do not want to SSE instructions
DFLAG += -DUSESSE

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: ffmpoly2-train ffmpoly2-predict

ffmpoly2-train: ffmpoly2_train.cpp ffmpoly2.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

ffmpoly2-predict: ffmpoly2_predict.cpp ffmpoly2.o timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

ffmpoly2.o: ffmpoly2.cpp ffm_poly2.h timer.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

timer.o: timer.cpp timer.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f ffmpoly2-train ffmpoly2-predict ffmpoly2.o timer.o
