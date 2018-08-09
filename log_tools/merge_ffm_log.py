#! /usr/bin/python

file_name = "mnist_ove.scale.tr_va.ffm.{fold}.va.{l}.{r}"
log_name = "mnist_ove.scale.tr_va.ffm.va.{l}.{r}.merge"
NF=6
COL=3
LSKIP=5

def merge_log(l, r, v):
    of = open(log_name.format(l=l, r=r, v=v), 'w')
    res = []
    for fold in range(1,NF+1):
        sf = open(file_name.format(fold=fold, l=l, r=r, v=v), 'r')
        sf_res = []
        for i, line in enumerate(sf):
            if i < LSKIP:
                continue
            if i == LSKIP:
                print(line)
            if len(line.strip().split()) >= COL:
                sf_res.append( float(line.strip().split()[COL-1]) )
                #print( (line, sf_res[-1]) )
        res.append(sf_res)
    for i in range(min(map(len,res))):
        total = 0.0
        for j in range(len(res)):
            total = total + res[j][i]
        total = total / float(NF)
        of.write("iter    {i}    {total}\n".format(i=i+1, total=total) )


def main():
    for l in ['1e-7', '0']:
        for r in ['0.4', '0.8', '1.6']:
            for v in ['8', '68', '571', '4747']:
                merge_log(l, r, v)

if __name__ == "__main__":
    main()
