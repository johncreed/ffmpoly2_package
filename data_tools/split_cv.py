#!/usr/bin/env python

import sys, random, os


def main():
    if len(sys.argv) != 3:
        print("./split_cv.py file num_fold")
        sys.exit(1)

    src_path = sys.argv[1]
    num_fold = int(sys.argv[2])
    f = open(src_path, 'r')
    random.seed(100)
    of_ls = []
    #create split file
    for i in range(1, num_fold+1):
        of_ls.append(open(src_path+'.{}.va'.format(i), 'w'))
    for line in f :
        idx = random.randint(1,num_fold) - 1
        of_ls[idx].write(line)

if __name__ == '__main__':
    main()
