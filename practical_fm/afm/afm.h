#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cmath>
#include <numeric>
#include <iomanip>

#include "omp.h"

typedef double AfmVal;
typedef unsigned long AfmIdx;

enum {FM_AGD=1, FM_ADAGRAD=5, AFM_ADAGRAD=6, FM_GD=10, AFM_ALS_PRIMAL=11};

using namespace std;

struct Parameter
{
    AfmIdx solver;
    AfmVal lambda_w;
    AfmVal lambda_U;
    AfmVal lambda_V;
    AfmIdx K;
    AfmIdx iters;
    AfmVal tol;
    AfmVal nu;
    AfmVal kappa;
    AfmVal std;
    AfmIdx cg_iters; // for Alternative (-s 11) only
    AfmVal eta; // for AdaGrad (-s 6) only
};

struct Node
{
    AfmIdx idx; 
    AfmVal val;
};

struct AfmData
{
    AfmData(): I(0), J(0) {};
    AfmIdx I;
    AfmIdx J;
    vector<AfmVal> labels;
    vector<Node*> indexes;
    vector<Node> nodes;
};

struct AfmModel
{
    AfmIdx solver;
    AfmIdx J;
    AfmIdx K;
    vector<AfmVal> w;
    vector<AfmVal> U;
    vector<AfmVal> V;
};

Parameter get_default_parameter();
void read_csr_problem(
    string filename,
    AfmData &data);
void read_csc_problem(
    string filename,
    AfmData &data);
void initialize_model(
    const AfmVal std,
    const AfmIdx solver,
    const AfmIdx K,
    const AfmIdx J,
    AfmModel &model);
//////////////////////
// AFM trainer
//////////////////////
void train_afm_adagrad(
    const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    AfmModel &model);
void train_afm_adagrad_no_self(
    const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    AfmModel &model);
void als_train_afm_primal(
    const AfmData &data,
    const AfmData &test,
    const Parameter param,
    const bool use_freq,
    const bool do_pcond,
    const bool do_gd,
    AfmModel &model);
void als_train_afm_primal_no_self(
    const AfmData &data,
    const AfmData &test,
    const Parameter param,
    const bool use_freq,
    const bool do_pcond,
    const bool do_gd,
    AfmModel &model);
//////////////////////
// FM trainer
//////////////////////
void train_fm_adagrad(
    const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    const bool drop_self,
    AfmModel &model);
void train_fm_gd(
    AfmData &data,
    AfmData &data_test,
    Parameter param,
    AfmModel &model);
void train_fm_als_gd(const AfmData &data,
    const AfmData &data_test,
    const Parameter param,
    const bool use_freq,
    AfmModel &model);
//////////////////////
// evaluating function
//////////////////////
AfmVal evaluate(
    const AfmData &data,
    const AfmModel &model,
    const bool drop_self);
