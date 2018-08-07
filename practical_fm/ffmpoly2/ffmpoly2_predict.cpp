//
// Created by Xing Tang on 2017/11/20.
//
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include "ffm_poly2.h"

using namespace std;
using namespace ffm_poly2;

struct Option{
    string test_path,model_path,output_path,pair_path;
};

string predict_help(){
    return string(
            "usage: ffm-predict test_file model_file pair_path output_path\n"
    );
}

Option parse_option(int argc, char **argv){
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 5)
        throw invalid_argument("cannot parse argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.pair_path = string(args[3]);
    option.output_path = string(args[4]);

    return option;
}

void predict(string test_path, string model_path, string output_path, string pair_path){
    int const kMaxLineSize = 10000000;
    
    FILE *f_in = fopen(test_path.c_str(), "r");
    ofstream f_out(output_path);
    vector<char> line(kMaxLineSize);
    
    ffm_poly2_model model =  ffm_poly2_load_model(model_path);
    ffm_double loss = 0;

    vector<ffm_poly2_node> X;

    ffm_int i = 0;

    while(fgets(line.data(),kMaxLineSize,f_in)){
        X.clear();
        char *y_char = strtok(line.data()," \t");
        ffm_float y = (atoi(y_char) >0 ) ? 1.0f : -1.0f;
        while(true){
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;
            ffm_poly2_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);
            X.push_back(N);
        }
        ffm_float y_bar = ffm_poly2_predict(X.data(),X.data()+X.size(),model,pair_path);
        loss -= y==1?log(y_bar):log(1-y_bar);
        f_out<<y_bar<<"\n";
        i++;
    }

    loss /= i;

    cout<< "logloss = "<< fixed << setprecision(5)<< loss << endl;

    fclose(f_in);
}

int main(int argc, char ** argv){
    Option option;
    try{
        option = parse_option(argc, argv);
    } catch (invalid_argument const &e){
        cout << e.what() << endl;
        return 1;
    }
    predict(option.test_path,option.model_path,option.output_path,option.pair_path);

    return 0;
}
