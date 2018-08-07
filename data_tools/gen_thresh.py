#! /usr/bin/python

import sys

def main():
    if len(sys.argv) != 2:
        print( "Usage: ./gen_thresh max_idx_num" )
        sys.exit(1)
    max_idx = float(sys.argv[1])
    print("max_idx: {}".format(max_idx))
    cur_v = 1.0
    v_list = []
    ratio = float(max_idx) ** 0.2
    v_list.append(int(cur_v) - 1)
    while cur_v < max_idx:
        cur_v = cur_v * ratio
        v_list.append(int(cur_v))
    res = ""
    for x in v_list:
        res += "{} ".format(str(x))
    print(res)


if __name__ == '__main__':
    main()
