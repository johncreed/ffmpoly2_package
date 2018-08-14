#! /usr/bin/python

import sys

file_name = "webspam_wc_normalized_unigram.tr_va.ffm.{fold}.va.{l}.{r}.{v}"
log_name = "webspam_wc_normalized_unigram.tr_va.ffm.va.{l}.{r}.{v}.merge"

file_name = "german-numer_scale.tr_va.ffm.{fold}.va.{k}.{l}.{r}.{v}"
log_name = "german-numer_scale.tr_va.ffm.va.{k}.{l}.{r}.{v}.merge"

NF=4
SKPL=6
COL=4

def merge_log(k ,l, r, v):
    of = open(log_name.format(k=k, l=l, r=r, v=v), 'w')
    res = []
    for fold in range(1,NF+1):
        sf = open(file_name.format(fold=fold, k=k, l=l, r=r, v=v), 'r')
        sf_res = []
        for i, line in enumerate(sf):
            if i <= SKPL-1:
                continue
            if i == SKPL:
                print(line.strip())
            if len(line.strip().split()) >= COL:
                sf_res.append( float(line.strip().split()[COL-1]) )
                #print(line, sf_res[-1])
        res.append(sf_res)
    print( min(map(len,res)) )
    for i in range(min(map(len,res))):
        total = 0.0
        for j in range(len(res)):
            total = total + res[j][i]
        total = total / float(NF)
        of.write("iter    {i}    {total}\n".format(i=i+1, total=total) )

ks = "6"
vs = "0 160 320 480 540 800"
rs = "0.05 0.0125 0.003125"
ls = "0 1e-5 1e-6 1e-7"

def main():
    for k in ks.strip().split():
        for l in ls.strip().split():
            for r in rs.strip().split():
                for v in vs.strip().split():
                    merge_log(k, l, r, v)

if __name__ == "__main__":
    main()
