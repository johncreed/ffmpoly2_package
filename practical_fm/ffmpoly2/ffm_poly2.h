//
// Created by Xing Tang on 2017/10/26.
//
#pragma once
#ifndef FFMPOLY2_FFM_POLY2_H
#define FFMPOLY2_FFM_POLY2_H

#include <string>

namespace ffm_poly2{
    using namespace std;

    typedef float ffm_float;
    typedef int ffm_int;
    typedef long long ffm_long;
    typedef double ffm_double;
    struct ffm_poly2_node
    {
        ffm_int f;// field字段
        ffm_int j;//feature字段
        ffm_float v;//value字段

    };


    struct ffm_poly2_model
    {
        ffm_int n; //特征数量
        ffm_int m; //field数量
        ffm_int k; //latent factor数量
        ffm_int p_num; //pair的数量
        ffm_float threshold;//pair的门限值
        ffm_float *W = nullptr;
        bool normalization;
        ~ffm_poly2_model();
    };

    struct ffm_poly2_parameter
    {
        ffm_float eta = 0.2;
        ffm_float ffm_lambda = 0.00002;
        ffm_float poly2_lambda = 0.0;
        ffm_int nr_iters = 15;
        ffm_long val_threshold = -1;
        ffm_int k = 4;
        bool auto_stop = false;
        bool normalization = true;
    };

    void ffm_poly2_read_problem_to_disk(string txt_path, string bin_path);
    void ffm_poly2_save_model(ffm_poly2_model &model, string path);
    ffm_poly2_model ffm_poly2_load_model(string path);
    ffm_float ffm_poly2_predict(ffm_poly2_node *begin, ffm_poly2_node *end, ffm_poly2_model &model, string pr_path);
    ffm_poly2_model ffm_poly2_train_on_disk(string tr_path, string va_path, string pr_path, ffm_poly2_parameter params);
}
#endif //FFMPOLY2_FFM_POLY2_H


