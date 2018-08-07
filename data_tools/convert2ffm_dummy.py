#!/usr/bin/env python3

import sys

def convert( f ) :
    with open(f+".ffm", 'w') as f_dst:
        for line in open(f):
            tokens = line.strip().split()
            label = tokens[0]
            if float(label) == 1:
                output = '1'
            else:
                output = '-1'
            for token in tokens[1:]:
                dim, val = token.split(':')
                output += ' 0:{dim}:{val}'.format(dim=dim, val=val)
            f_dst.write(output + '\n')


def main():
    argc = len(sys.argv)
    argv = sys.argv
    if argc == 1 :
        print ("./convert2ffm_dummy.py file1 file2 ... (Convert file to file.ffm)")
        sys.exit()

    for f in argv[1:] :
        convert(f)

    print( "Finish!" )

if __name__ == "__main__" :
    main()
