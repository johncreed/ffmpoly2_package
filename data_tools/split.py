#!/usr/bin/env python3

import argparse, sys, random

if len(sys.argv) != 3:
    print( "./split.py file_name ratio")
    print( "tr_size=ratio*file_size and te_size = (1-ratio)*file_size" )
    sys.exit(1)
file_name = sys.argv[1]
ratio = float(sys.argv[2])
random.seed(0)

with open(file_name+".tr", 'w') as f_tr, open(file_name+".te", 'w') as f_te:
    for line in open(file_name):
        if random.random() < ratio:
            f_tr.write(line)
        else:
            f_te.write(line)
