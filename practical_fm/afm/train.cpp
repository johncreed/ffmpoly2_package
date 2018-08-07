#include <iostream>
#include <string>
#include <stdexcept>

#include "afm.h"

#include <fenv.h>

AfmIdx nr_threads = 1;
AfmIdx subsample_rate = 1;
AfmIdx max_nt_iters = 1;
AfmVal nt_eps = 0.8;
bool is_relative_cg = false;

struct Option
{
    Parameter param;
    bool do_warm_start;
    bool use_freq;
    bool do_pcond;
    bool do_gd;
    bool drop_self;
    string data_path;
    string test_path;
};

bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

string train_help()
{
    return string(
    "usage: train [options] training_set_file test_set_file\n"
    "\n"
    "options:\n"
    "-lw <lambda_w>: set regularization coefficient on linear parameters (default 1)\n"
    "  Set a negative value to drop linear term.\n"
    "-lu <lambda_u>: set regularization coefficient on left latent basis (default 4)\n"
    "  Set a negative value to drop linear term.\n"
    "-lv <lambda_v>: set regularization coefficient on right latent basis (default 4)\n"
    "  Set a negative value to drop linear term.\n"
    "-s <solver>: set solver\n"
    "\t 1 -- FM AGD (no self-interaction)\n"
    "\t\t--freq: Use frequency-awared regularization.\n"
    "\t 5 -- FM AdaGrad\n"
    "\t\t--eta: Specify the initial learning rate.\n"
    "\t\t--freq: Use frequency-awared regularization.\n"
    "\t\t--ds: Drop self-interaction.\n"
    "\t 6 -- AFM AdaGrad\n"
    "\t\t--eta: Specify the initial learning rate.\n"
    "\t\t--freq: Use frequency-awared regularization.\n"
    "\t\t--ds: Drop self-interaction.\n"
    "\t10 -- FM GD\n"
    "\t11 -- AFM ALS primal solver\n"
    "\t\t--freq: Use frequency-awared regularization.\n"
    "\t\t--pcond: Use preconditioned CG.\n"
    "\t\t--gd: Use gradient rather than Newton direction.\n"
    "\t\t--ds: Drop self-interaction.\n"
    "\t\t--cg: Specify number of CG iterations (0 for relative cg stopping condtion).\n"
    "-k <dim>: set number of dimensions (default 4)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-e <tol>: set tolerance (default 0.001)\n"
    "-p <path>: set path to test set\n"
    "-u <nu>: set descent ratio in line search (default 0.01)\n"
    "-c <threads>: set number of cores (need -s 11)\n"
    "--sub <subset>: set ratio of subset (need -s 11)\n"
    "--std <std>: set the scale of initial model distribution (default 0.1)\n"
    "--warm: use one iteration SG-AdaGrad to do warm-start\n"
    );
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;
    option.do_warm_start = false;
    option.do_pcond = false;
    option.use_freq = false;
    option.do_gd = false;
    option.drop_self = false;
    option.param = get_default_parameter();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-lw") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify linear regularization\
                                        coefficient after -lw");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-lw should be followed by a number");
            option.param.lambda_w = atof(argv[i]);
        }
        else if(args[i].compare("-lu") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify left space\
                                        regularization coefficient\
                                        after -lu");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-lu should be followed by a number");
            option.param.lambda_U = atof(argv[i]);
        }
        else if(args[i].compare("-lv") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify right space\
                                        regularization coefficient\
                                        after -lv");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-lv should be followed by a number");
            option.param.lambda_V = atof(argv[i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify solver type after -s");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-s should be followed by a number");
            option.param.solver = atoi(argv[i]);

            if (option.param.solver != FM_AGD &&
                option.param.solver != FM_ADAGRAD &&
                option.param.solver != AFM_ADAGRAD &&
                option.param.solver != FM_GD &&
                option.param.solver != AFM_ALS_PRIMAL)
                throw invalid_argument("unsupported solver");
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify the number of\
                                        latent factors after -k");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-k should be followed by a number");
            option.param.K = atoi(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param.iters = atoi(argv[i]);
        }
        else if(args[i].compare("-e") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify tollerance after -e");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param.tol = atof(argv[i]);
        }
        else if(args[i].compare("--alpha") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify tollerance after --alpha");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            nt_eps = atof(argv[i]);
        }
        else if(args[i].compare("-u") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify descent ratio after\
                                        -u");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-u should be followed by a number");
            option.param.nu = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.test_path = string(args[i]);
        }
        else if(args[i].compare("--warm") == 0)
        {
            option.do_warm_start = true;
        }
        else if(args[i].compare("--freq") == 0)
        {
            option.use_freq = true;
        }
        else if(args[i].compare("--pcond") == 0)
        {
            option.do_pcond = true;
        }
        else if(args[i].compare("--gd") == 0)
        {
            option.do_gd = true;
        }
        else if(args[i].compare("--ds") == 0)
        {
            option.drop_self = true;
        }
        else if (args[i].compare("--cg") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify # of CG iterations after --cg");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("--cg should be followed by a number");
            option.param.cg_iters = atoi(argv[i]);
            if(option.param.cg_iters == 0)
                is_relative_cg = true;
        }
        else if (args[i].compare("--sub") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify # of subset ratio after --sub");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("--sub should be followed by a number");
            subsample_rate = atoi(argv[i]);
        }
        else if (args[i].compare("--nt") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify # of subset ratio after --sub");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("--nt should be followed by a number");
            max_nt_iters = atoi(argv[i]);
        }
        else if(args[i].compare("--std") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify initial range after --std");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("--std should be followed by a number");
            option.param.std = atof(argv[i]);
        }
        else if(args[i].compare("--eta") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to learning rate range after --eta");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("--eta should be followed by a number");
            option.param.eta = atof(argv[i]);
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw invalid_argument("training data not specified");
    option.data_path = string(args[i++]);

    return option;
}

int main(int argc, char *argv[])
{
    //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
    try
    {
        Option option = parse_option(argc, argv);
        AfmData data;
        AfmData data_test;
        AfmModel model;
        if (option.param.solver != FM_AGD)
        {
            read_csr_problem(option.data_path.c_str(), data);
            if (!option.test_path.empty())
                read_csr_problem(option.test_path.c_str(), data_test);
        }
        else
        {
            read_csc_problem(option.data_path.c_str(), data);
            if (!option.test_path.empty())
                read_csr_problem(option.test_path.c_str(), data_test);
        }

        initialize_model(option.param.std, option.param.solver,
                         option.param.K, data.J, model);

        if (option.do_warm_start)
        {
            cout << "Warm-start...";
            const AfmIdx iters = option.param.iters;
            option.param.iters = 1;
            train_afm_adagrad(data, data_test,
                option.param, option.use_freq, model);
            option.param.iters = iters;
            cout << "Warm-start is done." << endl;
        }
        switch (option.param.solver)
        {
            case FM_AGD:
                train_fm_als_gd(data, data_test, option.param,
                                option.use_freq, model);
                break;
            case FM_ADAGRAD:
                train_fm_adagrad(data, data_test, option.param,
                                 option.use_freq, option.drop_self, model);
                break;
            case AFM_ADAGRAD:
                train_afm_adagrad(data, data_test,
                    option.param, option.use_freq, model);
                break;
            case FM_GD:
                train_fm_gd(data, data_test, option.param, model);
                break;
            case AFM_ALS_PRIMAL:
                als_train_afm_primal(data, data_test, option.param,
                    option.use_freq, option.do_pcond, option.do_gd, model);
                break;
        }
    }
    catch(invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
