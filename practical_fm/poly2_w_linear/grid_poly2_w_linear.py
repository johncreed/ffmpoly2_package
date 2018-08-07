#!/usr/bin/env python

import os, sys, subprocess

# Path
log_folder = "/home/johncreed/logs/"
data_folder = ""
solver_path = ""
tr_path = ""
va_path = ""
pair_path = ""
final_tr_path = ""
final_te_path = ""
final_pair_path = ""
output_path = ""

def parse_argv():
    for i, x in enumerate(sys.argv):
        print("{} : {} ".format(i, x))
    if len(sys.argv) != 10:
        print ("Usage: ./grid_para.py data_folder solver_path tr_path va_path tr_pair_path final_tr_path final_te_path final_pair_path log_filename.")
        sys.exit("123")
    global data_folder, solver_path, tr_path, va_path, pair_path, final_tr_path, final_te_path, final_pair_path, output_path
    data_folder = sys.argv[1]
    solver_path = sys.argv[2]
    tr_path = os.path.join(data_folder, sys.argv[3])
    va_path = os.path.join(data_folder, sys.argv[4])
    pair_path = os.path.join(data_folder, sys.argv[5])

    final_tr_path =os.path.join( data_folder, sys.argv[6] )
    final_te_path =os.path.join( data_folder, sys.argv[7] )
    final_pair_path = os.path.join(data_folder, sys.argv[8])

    output_path = os.path.join(log_folder, sys.argv[9])


def st_logloss(out):
    last_logloss = 10000
    for each_iter in out.split('\n')[2:-1]:
        logloss = float(each_iter.split()[-2])
        last_logloss = logloss
    return last_logloss

def find_min_logloss(out):
    min_val = 10000
    min_iter = 10000
    start = 0
    for line in out.split('\n'):
        tk = line.split()
        if not start:
            if tk[0] == "iter" :
                start = 1
            continue
        print (tk)
        if len(tk) == 0:
            continue
        num_iter = float(tk[0])
        logloss = float(tk[2])
        if logloss < min_val:
            min_val = logloss
            min_iter = num_iter
    return min_iter, min_val

ratio_list = []

def rough_train( rough ):
    min_v = 1
    max_v = 11
    stp = (max_v - min_v) / 5
    cur_v = min_v
    v_list = [cur_v]
    while cur_v < max_v + 1:
        cur_v = cur_v + stp
        v_list.append(cur_v)
    if rough :
        return {"fl_list" : [0, 5 * 1e-6, 5 * 1e-4],
                "pl_list" : [0, 5 * 1e-6, 5 * 1e-4],
                "k_list" : [4],
                "t" : 3,
                "v_list" : v_list,
                "r_list" : [0.01, 0.05, 0.5],
                "s" : 10,
                "vp" : va_path,
                "p" : pair_path,
                }
    else:
        return {# 1e-1 1e-2 1e-5 1e-7
                "fl_list" : [10.0 / pow(10.0,x) for x in range(6)],
                "pl_list" : [100.0 / pow(10.0,x) for x in range(6)],
                #"k_list" : [4, 8, 16, 32, 64],
                "k_list" : [16],
                "t" : 50,
                #"v_list" : [x for x in range(60,100) if x % 8 == 0] ,
                "v_list" : v_list ,
                # 0.2 0.1
                "r_list" : [0.1 * pow(2.0 , x) for x in range(5)],
                #"r_list" : [0.01, 0.02, 0.05, 0.2, 0.5],
                "s" : 10,
                "vp" : va_path,
                "p" : pair_path,
                 }
    """
    fl, pl list 0.
    v_list range for each data:
    covtype : min 1 max 11
    ijcnn_split_by_cj: min 66 max 120
    ijcnn : min 66 max 96
    webspam : min 1 max 581
    """

def main():
    # Parse argument
    parse_argv()

    # Set up grid range
    grid_param_dict = rough_train(0)

    fl_list = grid_param_dict["fl_list"]
    pl_list = grid_param_dict["pl_list"]
    k_list = grid_param_dict["k_list"]
    t = grid_param_dict["t"]
    v_list = grid_param_dict["v_list"]
    r_list = grid_param_dict["r_list"]
    s = grid_param_dict["s"]
    vp = grid_param_dict["vp"]
    p = grid_param_dict["p"]

    # Start to grid
    min_logloss = 10000
    min_l = 0
    min_pl = 0
    min_k = 0
    min_r = 0

    for l in fl_list:
        for k in k_list:
            for r in r_list:
                cmd = "{run} -l {l} -k {k} -t {t} -r {r} -s {s} -p {vp} {tr}".format(run=solver_path, l = l, k = k, t = t , r = r, s = s, vp = vp, tr = tr_path)
                print (cmd)
                proc = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                out, err = proc.communicate()
                print ("-"*20)
                print (out)
                min_num_iter, min_logloss_in_out = find_min_logloss(out)
                if min_logloss_in_out < min_logloss:
                    min_logloss = min_logloss_in_out
                    min_l = l
                    min_k = k
                    min_r = r
                    min_t = min_num_iter
                print ("cur_logloss {} min_logloss {}".format( min_logloss_in_out, min_logloss))
    cmd = "{run} -l {l} -k {k} -t {t} -r {r} -s 5 -p {p} {tr}".format(run=solver_path, l = min_l, k = min_k, t = t, r = min_r, p = final_te_path, tr = final_tr_path)
    print ("Train all.")
    print (cmd)
    proc = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    out, err = proc.communicate()
    min_logloss_in_out = out #find_last_logloss(out)

    with open(output_path, 'w') as out:
        out.write('{0}\n'.format(min_logloss_in_out))
        min_num_iter, min_logloss = find_min_logloss(min_logloss_in_out)
        out.write("iter {i} min_logloss {min_loss}".format( i = min_num_iter, min_loss = min_logloss))



if __name__ == "__main__" :
    main()



