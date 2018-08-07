//
// Created by Xing Tang on 2017/11/1.
//

#pragma GCC diagnostic ignored "-Wunused-result"

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <numeric>
#include <map>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <unordered_set>
#include "ffm_poly2.h"
#include "timer.h"

#if defined USEOMP
#include <omp.h>
#endif


namespace ffm_poly2{
    namespace {
        using namespace std;
        typedef pair<ffm_int,ffm_int> index_pair;

        ffm_int const kBIN_SIZE = (1<<24) - 1;
        ffm_int const kALIGNByte = 4;
        ffm_int const kALIGN = kALIGNByte / sizeof(ffm_float);
        ffm_int const kCHUNK_SIZE = 10000000;
        ffm_int const kMaxLineSize = 1000000;

        struct indexHash{
            size_t operator()(const index_pair& rhs) const{
                return ((((size_t)rhs.first + rhs.second)*(rhs.first+rhs.second+1)>>1)+rhs.second) & kBIN_SIZE;
            }
        };

        inline ffm_int get_k_aligned(ffm_int k){
            return (ffm_int) ceil((ffm_float) k / kALIGN) * kALIGN;
        }

        //calculate the hash index of a and b;
        inline ffm_int calc_w_idx(ffm_int a, ffm_int b){
            return ((((ffm_long)a + b)*(a+b+1)>>1)+b) & kBIN_SIZE;
        }

        inline ffm_float wTx(ffm_poly2_node *begin, ffm_poly2_node *end, ffm_float r,
                             ffm_poly2_model &model, unordered_set<index_pair,indexHash> &p_set,
                             ffm_float kappa=0,
                             ffm_float eta = 0, ffm_float ffm_lambda = 0,
                             ffm_float poly_lambda = 0, bool do_update = false){

            ffm_int ffm_align0 = 2 * get_k_aligned(model.k);
            ffm_int ffm_align1 = model.m * ffm_align0;
            ffm_int w_align = 2 * model.n * model.m * get_k_aligned(model.k);

            ffm_float t = 0;

            for(ffm_poly2_node *N1 = begin; N1 != end; N1++){
                ffm_int j1 = N1->j;
                ffm_int f1 = N1->f;
                ffm_float v1 = N1->v;

                if(j1 >=  model.n || f1 >= model.m)
                    continue;
                for(ffm_poly2_node *N2 = N1+1; N2 != end; N2++){
                    ffm_int j2 = N2->j;
                    ffm_int f2 = N2->f;
                    ffm_float v2 = N2->v;

                    if(j2 >= model.n || f2 >= model.m)
                        continue;

                    index_pair p(max(j1,j2),min(j1,j2));


                    if(p_set.find(p) != p_set.end()) {
                        ffm_int hash_val = calc_w_idx(p.first,p.second);
                        ffm_float *w = model.W + w_align + hash_val*2;
                        ffm_float v = v1*v2*r;
                        if(do_update){
                            ffm_float *wg = w + 1;
                            ffm_float g = poly_lambda * (*w)+ kappa * v;
                            *wg += g*g;
                            *w -= (eta/sqrt(*wg))*g;
                        }
                        else
                            t += (*w)*v;
                    }
                    else{
                        ffm_float *w1 = model.W + j1*ffm_align1 + f2*ffm_align0;
                        ffm_float *w2 = model.W + j2*ffm_align1 + f1*ffm_align0;

                        ffm_float v = v1*v2*r;
                        if(do_update){
                            ffm_float *wg1 = w1 + kALIGN;
                            ffm_float *wg2 = w2 + kALIGN;
                            for(ffm_int d = 0; d<ffm_align0; d+=kALIGN*2){
                                ffm_float g1 = ffm_lambda * w1[d] + kappa*w2[d]*v;
                                ffm_float g2 = ffm_lambda * w2[d] + kappa*w1[d]*v;

                                wg1[d] += g1 * g1;
                                wg2[d] += g2 * g2;

                                w1[d] -= eta /sqrt(wg1[d]) * g1;
                                w2[d] -= eta /sqrt(wg2[d]) * g2;
                            }
                        }else{
                            for(ffm_int d=0; d < ffm_align0; d+= kALIGN*2)
                                t += w1[d] *w2[d] *v;
                        }
                    }

                }
            }
            return t;
        }

        ffm_long get_w_size(ffm_poly2_model &model){
            ffm_int k_aligned = get_k_aligned(model.k);
            return (ffm_long) (model.n * model.m * k_aligned + kBIN_SIZE + 1) * 2;
        }

        ffm_float* malloc_aligned_float(ffm_long size){
            void *ptr;
            ptr = malloc(size* sizeof(ffm_float));

            return (ffm_float *) ptr;
        }
        //initialize model
        ffm_poly2_model init_model(ffm_int n, ffm_int m, ffm_int pair_num, ffm_float threshold_val,ffm_poly2_parameter param){
            ffm_poly2_model model;
            model.n = n;
            model.m = m;
            model.threshold = threshold_val;
            model.p_num = pair_num;
            model.k = param.k;
            model.W = nullptr;
            model.normalization = param.normalization;

            ffm_int k_aligned = get_k_aligned(model.k);

            model.W = malloc_aligned_float((ffm_long)(n*m*k_aligned+kBIN_SIZE + 1)*2);

            ffm_float coef = 1.0f/sqrt(model.k);
            ffm_float *w = model.W;

            default_random_engine generator;
            uniform_real_distribution<ffm_float> distribution(0.0,1.0);

            for (ffm_int j = 0; j < model.n ; j++) {
                for(ffm_int f = 0; f < model.m; f++){
                    for(ffm_int d = 0; d < k_aligned;){
                        for(ffm_int i = 0; i < kALIGN; i++,w++,d++){
                            w[0] = (d<model.k)?coef*distribution(generator) : 0.0;
                            w[kALIGN] = 1;
                        }
                        w += kALIGN;
                    }
                }

            }
            for(ffm_int i = 0; i < kBIN_SIZE + 1 ; i++){
                for(ffm_int j = 0; j<kALIGN; j++, w++){
                    w[0] = 0;
                    w[kALIGN] = 1;
                }
                w += kALIGN;
            }
            return model;
        }

        //load the "serialized" problem from disk;
        struct disk_problem_meta{
            ffm_int n = 0;//max feature index number
            ffm_int m = 0;//max field number
            ffm_int l = 0;//equal to num_blocks
            ffm_int num_blocks = 0;//the number of write chunks
            ffm_long B_pos = 0;//the bin file write position
            uint64_t hash1;
            uint64_t hash2;
        };

        // hash block with size kCHUNK_SIZE
        uint64_t hashfile(string txt_path, bool one_block = false){
            ifstream f(txt_path,ios::ate|ios::binary);
            if(f.bad())
                return 0;
            ffm_long end = (ffm_long) f.tellg();
            f.seekg(0, ios::beg);
            assert(static_cast<int>(f.tellg()) == 0);

            uint64_t magic = 90359;
            for(ffm_long pos = 0; pos < end;){
                ffm_long next_pos =  min(pos+kCHUNK_SIZE, end);
                ffm_long size = next_pos - pos;
                vector<char> buffer(kCHUNK_SIZE);
                f.read(buffer.data(), size);

                ffm_int i = 0;
                while(i < size - 8){
                    uint64_t x = *reinterpret_cast<uint64_t *> (buffer.data() + i);
                    magic = ((magic + x) * (magic + x +1) >> 1) + x;
                    i += 8;
                 }

                for(;i <  size; i++){
                    char x = buffer[i];
                    magic = ((magic + x) * (magic + x +1)>>1) + x;
                }
                pos = next_pos;
                if(one_block)
                    break;
            }
            return magic;
        }

        //define the problem on the disk
        struct problem_on_disk{
            disk_problem_meta meta;
            vector<ffm_float> Y;
            vector<ffm_float> R;
            vector<ffm_long> P;
            vector<ffm_poly2_node> X;
            vector<ffm_long> B;

            problem_on_disk(string path){
                f.open(path,ios::in | ios::binary);
                if(f.good()){
                    f.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
                    f.seekg(meta.B_pos);
                    B.resize(meta.num_blocks);
                    f.read(reinterpret_cast<char*>(B.data()), sizeof(ffm_long)*meta.num_blocks);
                }
            }

            int load_block(int block_index){
                if(block_index >= meta.num_blocks)
                    assert(false);
                f.seekg(B[block_index]);

                ffm_int l;
                f.read(reinterpret_cast<char*>(&l), sizeof(int));

                Y.resize(l);
                f.read(reinterpret_cast<char *>(Y.data()), sizeof(ffm_float) * l);

                R.resize(l);
                f.read(reinterpret_cast<char *>(R.data()), sizeof(ffm_float) * l);

                P.resize(l+1);
                f.read(reinterpret_cast<char *>(P.data()), sizeof(ffm_long) *(l+1));

                X.resize(P[l]);
                f.read(reinterpret_cast<char *>(X.data()), sizeof(ffm_poly2_node) * P[l]);

                return l;
            }

            bool is_empty(){
                return meta.l == 0;
            }
        private:
            ifstream f;
        };

        //compute the pair number
        ffm_float generate_pair(string bin_path,string pair_path){

            problem_on_disk tr(bin_path);
            ofstream f_pair(pair_path, ios::out);
            vector<ffm_int> block_order(tr.meta.num_blocks);
            iota(block_order.begin(),block_order.end(),0);
            ffm_long pos = 0;
            ffm_long neg = 0;
            map<index_pair,ffm_long> pairMap;

            for(auto blk:block_order){
                ffm_int in_num = tr.load_block(blk);
                for(ffm_int i = 0; i < in_num; i++){
                    if(tr.Y[i] == 1.0f)
                        pos ++;
                    else
                        neg ++;
                    for( ffm_long iter_a = tr.P[i] ; iter_a != tr.P[i+1]; iter_a++ ){
                        ffm_poly2_node trN_a = tr.X[iter_a];
                        for(ffm_long iter_b = iter_a+1; iter_b != tr.P[i+1]; iter_b++ ){
                            ffm_poly2_node trN_b = tr.X[iter_b];
                            index_pair p(max(trN_a.j,trN_b.j),min(trN_a.j,trN_b.j));

                            map<index_pair,ffm_long>::iterator iter = pairMap.find(p);

                            if(iter != pairMap.end())
                                pairMap[p] += 1;
                            else
                                pairMap[p] = 1;
                        }
                    }
                }
            }
            f_pair<<"positive:"<<pos<<"\t"<<"negative:"<<neg<<endl;
            for(map<index_pair,ffm_long>::iterator iter = pairMap.begin(); iter!= pairMap.end(); iter++)
            {
                index_pair p = iter->first;
                ffm_long count = iter->second;
                f_pair <<p.first<<","<<p.second<<"\t"<<count<<"\n";
            }
            f_pair.close();
            ffm_float ratio = (ffm_float)pos/(pos+neg);
            return (ffm_float)1536.0*(1-ratio)/(ratio);
        }

        unordered_set<index_pair,indexHash> get_pair(string pr_path, ffm_float threshold, ffm_int &pair_num) {
            ifstream pr_fi(pr_path, ios::in);
            if (!pr_fi.is_open())
                throw;
            string s;
            unordered_set<index_pair,indexHash> p_set;
            while (getline(pr_fi, s)) {
                size_t pos = s.find("\t");
                if (pos == string::npos)
                    throw;
                string pair_str = s.substr(0, pos);
                ffm_long count = atoi(s.substr(pos + 1, s.size()).c_str());
                if (count > threshold) {
                    pos = pair_str.find(",");
                    if (pos == string::npos)
                        throw;
                    ffm_int left = atoi(pair_str.substr(0, pos).c_str());
                    ffm_int right = atoi(pair_str.substr(pos + 1, pair_str.size()).c_str());
                    index_pair p(max(left, right), min(left,right));
                    p_set.insert(p);
                    pair_num += 1;
                }
            }
            return p_set;
        }
        //"serialize" problem
        void txt2bin(string txt_path, string bin_path){
            cout << "Warning : each instance length must less then " << kMaxLineSize << ", please check your txt file with cmd \"wc -L txt_file.\" if necessary." << endl;
            
            FILE *f_txt = fopen(txt_path.c_str(),"r");
            if(f_txt == nullptr)
                throw;

            ofstream f_bin(bin_path, ios::out | ios::binary);

            vector<char> line(kMaxLineSize);

            ffm_long p = 0;
            disk_problem_meta meta;

            vector<ffm_float> Y;//label
            vector<ffm_float> R;//scale value
            vector<ffm_long> P(1,0);//an instance contain start point.
            vector<ffm_poly2_node> X;//node
            vector<ffm_long> B;//every time the position where the bin file write.

            auto write_chunk = [&](){
                B.push_back((ffm_long)f_bin.tellp());
                ffm_int l = Y.size();
                ffm_long nnz = P[l];
                meta.l += l;

                f_bin.write(reinterpret_cast<char *>(&l), sizeof(ffm_int));
                f_bin.write(reinterpret_cast<char *>(Y.data()), sizeof(ffm_float)*l);
                f_bin.write(reinterpret_cast<char *>(R.data()), sizeof(ffm_float)*l);
                f_bin.write(reinterpret_cast<char *>(P.data()), sizeof(ffm_long)*(l+1));
                f_bin.write(reinterpret_cast<char *>(X.data()), sizeof(ffm_poly2_node) * nnz);

                Y.clear();
                R.clear();
                P.assign(1,0);
                X.clear();
                p=0;
                meta.num_blocks++;

            };

            f_bin.write(reinterpret_cast<char*> (&meta), sizeof(disk_problem_meta));
            //for one line extract information
            while(fgets(line.data(),kMaxLineSize,f_txt)){
                char *y_char = strtok(line.data()," \t");
                ffm_float y = (atoi(y_char) >0 ) ? 1.0f : -1.0f;
                ffm_float scale = 0;
                for(;;p++){
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

                    meta.m = max(meta.m,N.f + 1);
                    meta.n = max(meta.n,N.j + 1);

                    scale += N.v*N.v;
                }

                scale = 1.0/scale;
                Y.push_back(y);
                R.push_back(scale);
                P.push_back(p);

                if(X.size() > (size_t) kCHUNK_SIZE)
                    write_chunk();
            }
            write_chunk();
            write_chunk();//write a dummy empty chunk in order to know where the EOF is
            assert(meta.num_blocks == (ffm_int)B.size());
            meta.B_pos = f_bin.tellp();
            f_bin.write(reinterpret_cast<char *> (B.data()), sizeof(ffm_long)* B.size());
            fclose(f_txt);
            meta.hash1 = hashfile(txt_path,true);
            meta.hash2 = hashfile(txt_path,false);

            f_bin.seekp(0,ios::beg);
            f_bin.write(reinterpret_cast<char *>(&meta), sizeof(disk_problem_meta));

        }

        bool check_same_txt_bin(string txt_path, string bin_path){
            ifstream f_bin(bin_path, ios::binary|ios::ate);

            if(f_bin.tellg() < (ffm_long) sizeof(disk_problem_meta)){
                return false;
            }
            disk_problem_meta meta;
            f_bin.seekg(0,ios::beg);
            f_bin.read(reinterpret_cast<char *>(& meta), sizeof(disk_problem_meta));
            if(meta.hash1 != hashfile(txt_path, true))
                return false;
            if(meta.hash2 != hashfile(txt_path, false))
                return false;
            return true;
        }

    }

    ffm_poly2_model::~ffm_poly2_model() {
        if(W != nullptr) {
            free(W);
            W = nullptr;
        }
    }

    void ffm_poly2_read_problem_to_disk(string txt_path,string bin_path){
        bool same_file = check_same_txt_bin(txt_path,bin_path);

        if(same_file){
            cout << "Binary file found with same content as txt file. Skip converting text to binary format"<<endl;
        }else{
            cout<<"Binary file NOT found. Convert text file to binary file" << flush;
            txt2bin(txt_path,bin_path);
        }
    }

    ffm_poly2_model ffm_poly2_train_on_disk(string tr_path, string va_path, string pr_path,ffm_poly2_parameter param){
        problem_on_disk tr(tr_path);
        problem_on_disk va(va_path);

        ffm_float threshold_val;

        ifstream pr_fi;
        pr_fi.open(pr_path.c_str(), ios::in);
        if(!pr_fi){
            threshold_val = generate_pair(tr_path,pr_path);
        } 
        else{
            string FL;
            getline(pr_fi, FL);
            // pos_num is the count of y == +1, vice versa.
            ffm_long pos_num = atoi(FL.substr(FL.find(":")+1, FL.find("\t")).c_str());
            ffm_long neg_num = atoi(FL.substr(FL.find(":", FL.find("\t"))+1,FL.size()).c_str());
            cout << "Warning : the auto-generated threshold_val process is questionable." << endl;
            ffm_float ratio = (ffm_float)pos_num/(pos_num+neg_num);
            threshold_val = (ffm_float)1536.0*(1-ratio)/(ratio);
            pr_fi.close();
        }

        if(param.val_threshold != -1)
            threshold_val = param.val_threshold;
        cout<<"threshold value:"<<threshold_val<<endl;

        ffm_int pair_num = 0;

        unordered_set<index_pair,indexHash> p_set = get_pair(pr_path,threshold_val,pair_num);

        cout<<"pair number:"<<pair_num<<endl;

        ffm_poly2_model model = init_model(tr.meta.n,tr.meta.m,pair_num,threshold_val,param);

        bool auto_stop = param.auto_stop && !va_path.empty();

        ffm_long w_size = get_w_size(model);
        vector<ffm_float> prev_W(w_size,0);
        ffm_double best_va_loss = numeric_limits<ffm_double>::max();
        
        cout.width( 5 );
        cout << "";
        cout.width( 15 );
        cout<<"tr_logloss";
        if(!va_path.empty()){
            cout.width(15);
            cout << "va_logloss";
        }
        cout.width(15);
        cout << "accuracy";
        cout.width(10);
        cout << "tr_time";
        cout << endl;
        Timer timer;

        auto one_epoch = [&] (problem_on_disk &prob, bool do_update){
            ffm_double loss = 0;
            ffm_double accuracy = 0.0;

            vector<ffm_int> outer_order(prob.meta.num_blocks);
            iota(outer_order.begin(), outer_order.end(), 0);
            random_shuffle(outer_order.begin(),outer_order.end());
            for(auto blk:outer_order){
                ffm_int l = prob.load_block(blk);
                vector<ffm_int> inner_oder(l);
                iota(inner_oder.begin(), inner_oder.end(),0);
                random_shuffle(inner_oder.begin(), inner_oder.end());

#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:loss, accuracy)
#endif
                for(ffm_int ii = 0; ii < l; ii++){
                    ffm_int i = inner_oder[ii];
                    //timer.tic();
                    ffm_float y = prob.Y[i];
                    ffm_poly2_node *begin = &prob.X[prob.P[i]];
                    ffm_poly2_node *end = &prob.X[prob.P[i+1]];
                    ffm_float r = param.normalization ? prob.R[i] : 1;
                    ffm_double t = wTx(begin,end,r,model,p_set);
                    ffm_double kappa;

                    if( -y*t > 0 ){
                      ffm_double expyt = exp(y*t);
                      loss += -y*t + log1p(expyt); // log(1 + e^x) = x + log(1 + e^-x)
                      kappa = -y / (1 + expyt);
                    }
                    else{
                      ffm_double expnyt = exp(-y*t);
                      loss += log1p(expnyt);
                      kappa = -y*expnyt/(1+expnyt);
                      accuracy += 1;
                    }

                    if(do_update)
                          wTx(begin,end,r,model,p_set,kappa,param.eta,param.ffm_lambda,param.poly2_lambda,true);

                    //timer.toc();
                    //cout<<"one instance:"<<timer.get()<<endl;
                }
            }

            pair<ffm_double, ffm_double> res(loss/prob.meta.l, accuracy/prob.meta.l);

            return res;
        };

        for(ffm_int iter = 1; iter <= param.nr_iters; iter ++){
            timer.tic();
            pair<ffm_double, ffm_double> tr_res = one_epoch(tr,true);
            ffm_double tr_loss = tr_res.first;
            //ffm_double tr_accuracy = tr_res.second;
            //ffm_double tr_loss = one_epoch(tr,true);
            timer.toc();
           
            cout.width(5); 
            cout << "iter ";
            cout << iter;
            cout.width(15);
            cout << fixed << setprecision(5) << tr_loss;

            if(!va.is_empty()){
                pair<ffm_double, ffm_double> va_res = one_epoch(va, false);
                ffm_double va_loss = va_res.first;
                ffm_double va_accuracy = va_res.second;
                //ffm_double va_loss = one_epoch(va, false);

                cout.width(15);
                cout << fixed << setprecision(5) << va_loss;

                cout.width(15);
                cout << fixed << setprecision(5) << va_accuracy;

                if(auto_stop){
                    if(va_loss > best_va_loss){
                        memcpy(model.W, prev_W.data(), w_size* sizeof(ffm_float));
                        cout << endl << "Auto-stop. Use model at" << iter-1 <<"th iteration."<<endl;
                        break;
                    }else{
                        memcpy(prev_W.data(),model.W, w_size * sizeof(ffm_float));
                        best_va_loss = va_loss;
                    }
                }
            }
            else
            {
                ostringstream ostr;
                ostr<<"model_"<<iter<<endl;
                string model_path = ostr.str();
                ffm_poly2_save_model(model, model_path);
            }
            cout.width(10);
            cout << fixed << setprecision(1) << timer.get() << endl;
        }
        return model;
    }

    void ffm_poly2_save_model(ffm_poly2_model &model, string path){
        ofstream f_out(path, ios::out | ios::binary);
        f_out.write(reinterpret_cast<char*> (&model.n), sizeof(ffm_int));
        f_out.write(reinterpret_cast<char*> (&model.m), sizeof(ffm_int));
        f_out.write(reinterpret_cast<char*> (&model.k), sizeof(ffm_int));
        f_out.write(reinterpret_cast<char*> (&model.p_num), sizeof(ffm_int));
        f_out.write(reinterpret_cast<char*> (&model.threshold), sizeof(ffm_float));
        f_out.write(reinterpret_cast<char*> (&model.normalization), sizeof(bool));

        ffm_long w_size = get_w_size(model);
        //model.W = malloc_aligned_float(w_size);
        for(ffm_long offset = 0; offset < w_size;){
            ffm_long next_offset = min(w_size, offset + (ffm_long) sizeof(ffm_float) *kCHUNK_SIZE);
            ffm_long size = next_offset - offset;
            f_out.write(reinterpret_cast<char*>(model.W + offset), sizeof(ffm_float)*size);
            offset = next_offset;
        }

    }

    ffm_poly2_model ffm_poly2_load_model(string path){
        ifstream f_in(path, ios::out | ios::binary);
        ffm_poly2_model model;
        f_in.read(reinterpret_cast<char*> (&model.n), sizeof(ffm_int));
        f_in.read(reinterpret_cast<char*> (&model.m), sizeof(ffm_int));
        f_in.read(reinterpret_cast<char*> (&model.k), sizeof(ffm_int));
        f_in.read(reinterpret_cast<char*> (&model.p_num), sizeof(ffm_int));
        f_in.read(reinterpret_cast<char*> (&model.threshold), sizeof(ffm_float));
        f_in.read(reinterpret_cast<char*> (&model.normalization), sizeof(bool));

        ffm_long w_size = get_w_size(model);
        model.W = malloc_aligned_float(w_size);
        for(ffm_long offset = 0; offset < w_size;){
            ffm_long next_offset = min(w_size, offset + (ffm_long) sizeof(ffm_float) *kCHUNK_SIZE);
            ffm_long size = next_offset - offset;
            f_in.read(reinterpret_cast<char*>(model.W + offset), sizeof(ffm_float)*size);
            offset = next_offset;
        }
        return model;
    }

    ffm_float ffm_poly2_predict(ffm_poly2_node *begin, ffm_poly2_node *end, ffm_poly2_model &model, string pr_path){
        ffm_float r = 1;
        if(model.normalization){
            r = 0;
            for(ffm_poly2_node *N = begin; N != end; N++)
                r += N->v *N->v;
            r = 1/r;
        }
        ffm_int pair_num = 0;
        unordered_set<index_pair,indexHash> p_set = get_pair(pr_path,model.threshold,pair_num);
        ffm_float t = wTx(begin,end,r,model,p_set);

        return 1/(1 + exp(-t));
    }

}
