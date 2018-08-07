#!/usr/bin/env python

import sys, random, os
#concatenate to train file
def main():
    if len(sys.argv) != 3:
        print("./split_cv.py file num_fold")
        sys.exit(1)
    src_path = sys.argv[1]
    num_fold = int(sys.argv[2])

    for i in range(1, num_fold+1):
        cmd = "cat"
        tr_f = src_path+'.{}.tr'.format(i)
        for j in range(1, num_fold+1):
            if i == j :
                continue
            else:
                va_f = src_path+'.{}.va'.format(j)
                cmd = '{} {}'.format(cmd, va_f)
        cmd = "{} > {}".format(cmd, tr_f)
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    main()
