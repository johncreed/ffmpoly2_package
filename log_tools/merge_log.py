#! /usr/bin/python

import sys

file_name = "ijcnn1.tr_va.ffm.{fold}.va.{l}.{r}.{v}"
log_name = "ijcnn1.tr_va.ffm.va.{l}.{r}.{v}.merge"
NF=4

def merge_log(l, r, v):
    of = open(log_name.format(l=l, r=r, v=v), 'w')
    res = []
    for fold in range(1,NF+1):
        sf = open(file_name.format(fold=fold, l=l, r=r, v=v), 'r')
        sf_res = []
        start=0
        for i, line in enumerate(sf):
            tokens = line.strip().split()
            if len(tokens) >= 2:
                if tokens[1] == "tr_logloss" or tokens[0] == "tr_logloss":
                    start=1
                    continue
            if start == 0:
                continue
            if start == 1:
                print(line)
                start = start + 1
            if len(line.strip().split()) >= 4:
                if tokens[0] == "iter":
                    sf_res.append( float(tokens[3]) )
                else:
                    sf_res.append( float(tokens[2]))
                print(line, sf_res[-1])
        res.append(sf_res)
    for i in range(min(map(len,res))):
        total = 0.0
        for j in range(len(res)):
            total = total + res[j][i]
        total = total / float(NF)
        of.write("iter    {i}    {total}\n".format(i=i+1, total=total) )


def main():
    for l in ['0', '1e-7', '1e-6', '1e-5']:
        for r in ['3.2', '1.6', '0.8', '0.4', '0.2', '0.1']:
            for v in ['0', '3', '32', '336', '3453', '35411', '113397' ]:
                merge_log(l, r, v)

if __name__ == "__main__":
    main()
