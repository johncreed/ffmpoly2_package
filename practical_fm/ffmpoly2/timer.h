//
// Created by Xing Tang on 2017/11/17.
//

#ifndef FFMPOLY2_TIMER_H
#define FFMPOLY2_TIMER_H

#endif //FFMPOLY2_TIMER_H
#include <chrono>

class Timer
{
public:
    Timer();
    void reset();
    void tic();
    float toc();
    float get();
private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;
};