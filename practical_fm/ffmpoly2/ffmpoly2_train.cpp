#pragma GCC diagnostic ignored "-Wunused-result"

#include <algorithm>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>

#include "ffm_poly2.h"
#if defined USEOMP
#include <omp.h>
#endif

using namespace std;
using namespace ffm_poly2;

string train_help()
{
    return string(
            "usage:ffm_poly2 [options] training_set_file [model file]\n"
            "\n"
            "options:\n"
            "-fl <ffm_lambda>:set ffm part regularization parameter (default 0.00002)\n"
            "-pl <poly2_lambda>:set regularization poly2 part regularization parameter (default 0.0)\n"
            "-k <factor>:set number of latent factors (default 4)\n"
            "-t <iteration>:set number of iterations (default 15)\n"
            "-v <threshold value>:set the threshold value and -1 generate the value auto (default -1)\n"
            "-r <eta>:set learning rate (default 0.2)\n"
            "-s <nr_threads>:set number of threads (default 1)\n"
            "-vp <validation path> set path to the validation set\n"
            "-p  <pair path> set path to store the pair count\n"
            "--quiet:quiet model(no output)\n"
            "--no-norm:disable instance-wise normalization\n"
            "--auto-stop:stop at the iteration that achieves the best validation loss (must be used with -vp)\n"
    );
}

struct Option{
    string tr_path;
    string va_path;
    string pr_path;
    string model_path;
    ffm_poly2_parameter param;
    bool quiet = false;
    ffm_int nr_threads = 1;
};

string basename(string path){

    const char *ptr = strrchr(&*path.begin(),'/');//get the last occurance of '/'
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

Option parse_option(int argc, char** argv){
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    if(argc == 1)
        throw invalid_argument(train_help());

    Option opt;

    ffm_int i =1;

    for(;i < argc; i++) {
        if (args[i].compare("-t") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            opt.param.nr_iters = atoi(args[i].c_str());
            if (opt.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        } else if (args[i].compare("-k") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            opt.param.k = atoi(args[i].c_str());
            if (opt.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        } else if (args[i].compare("-r") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            opt.param.eta = atof(args[i].c_str());
            if (opt.param.eta <= 0)
                throw invalid_argument("learning rate eta should be greater than zero");
        } else if (args[i].compare("-fl") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify ffm lambda after -lf");
            i++;
            opt.param.ffm_lambda = atof(args[i].c_str());
            if (opt.param.ffm_lambda < 0)
                throw invalid_argument("ffm regularization cost should not be smaller than zero");
        } else if (args[i].compare("-pl") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify poly2 lambda after -pl");
            i++;
            opt.param.poly2_lambda = atof(args[i].c_str());
            if (opt.param.poly2_lambda < 0)
                throw invalid_argument("poly2 regularization cost should not be smaller than zero");
        } else if (args[i].compare("-s") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.nr_threads = atoi(args[i].c_str());
            if (opt.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        } else if(args[i].compare("-v") == 0){
            if(i == argc - 1)
                throw invalid_argument("need to specify the value of threshold after -v");
            i++;
            opt.param.val_threshold = atoi(args[i].c_str());
            if(opt.param.val_threshold < -1)
                throw invalid_argument("value of threshold should not be less than -1");
        } else if (args[i].compare("--no-norm") == 0) {
            opt.param.normalization = false;
        } else if (args[i].compare("--quiet") == 0) {
            opt.quiet = true;
        } else if (args[i].compare("-vp") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify validation path after -vp");
            i++;
            opt.va_path = args[i];
        } else if (args[i].compare("-p") == 0){
            if(i == argc - 1)
                throw invalid_argument("need to specify pair path after -p");
            i++;
            opt.pr_path = args[i];
        }
        else if(args[i].compare("--auto-stop") == 0){
            opt.param.auto_stop = true;
        } else {
            break;
        }
    }

    if (i != argc - 2 && i != argc-1)
        throw invalid_argument("cannot parse command\n");

    opt.tr_path = args[i];
    i++;

    if(i<argc){
        opt.model_path = string(args[i]);
    } else if(i == argc)
        opt.model_path = basename(opt.tr_path) + ".model";
    else
        throw invalid_argument("cannot parse argument\n");
    return opt;
}

int train_on_disk(Option opt){
    string tr_bin_path = basename(opt.tr_path) + ".bin";
    string va_bin_path = opt.va_path.empty()? "" : basename(opt.va_path) + ".bin";

    if(!opt.va_path.empty())
        ffm_poly2_read_problem_to_disk(opt.va_path,va_bin_path);
    ffm_poly2_read_problem_to_disk(opt.tr_path, tr_bin_path);

    ffm_poly2_model model = ffm_poly2_train_on_disk(tr_bin_path.c_str(), va_bin_path.c_str(), opt.pr_path.c_str(), opt.param);

    ffm_poly2_save_model(model, opt.model_path);

    return 0;
}

int main(int argc, char **argv) {
    Option opt;
    try{
        opt = parse_option(argc, argv);
    } catch (invalid_argument &e){
        cout<< e.what() << endl;
        return 1;
    }

    if(opt.quiet)
        cout.setstate(ios_base::badbit);

    if(opt.param.auto_stop && opt.va_path.empty()){
        cout <<"To use auto-stop, you need to assign a validation set" << endl;
        return 1;
    }

#if defined USEOMP
    omp_set_num_threads(opt.nr_threads);
#endif

    train_on_disk(opt);

    return 0;

}