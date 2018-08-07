#!/usr/bin/env python3

import sys
import math

mapping_dict = {}
max_feat_idx = 55

def build_dict():
    global mapping_dict
    idd = 1
    for i in range(1, max_feat_idx):
        for j in range(i+1, max_feat_idx):
            mapping_dict[('%d'%i, '%d'%j)] = idd
            idd += 1
    print(idd)


def convert( f ) :
    global mapping_dict
    with open(f+".poly2", 'w') as f_dst:
        for line in open(f):
            tuples = []
            tokens = line.strip().split()
            label = tokens[0]
            if float(label) == 1:
                output = '1'
            else:
                output = '-1'
            norm_base = 0
            for i in range(1, len(tokens)):
                dim1, val1 = tokens[i].split(':')
                norm_base += float(val1)*float(val1)
            norm_base = 1/norm_base
            for i in range(1, len(tokens)):
                dim1, val1 = tokens[i].split(':')
                for j in range(i+1, len(tokens)):
                    dim2, val2 = tokens[j].split(':')
                    combin_index = (dim1, dim2)
                    combin_dim = mapping_dict[combin_index]
                    combin_val = float(val1)*float(val2)*norm_base
                    if not combin_val:
                        continue
                    tuples.append((combin_dim, combin_val))
            tuples = [(i[0], i[1]) for i in tuples]
            output = output +  ' ' + ' '.join(["{0}:{1}".format(i[0], i[1]) for i in tuples])
            f_dst.write(output + '\n')


def main():
    build_dict()
    argc = len(sys.argv)
    argv = sys.argv
    if argc == 1 :
        print ("./convert2poly2.py file1 file2 ... (Convert file to file.poly2)")
        sys.exit()

    for f in argv[1:] :
        convert(f)

    print( "Finish!" )

if __name__ == "__main__" :
    main()
