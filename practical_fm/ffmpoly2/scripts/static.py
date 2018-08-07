#! /usr/bin/python3
import sys

def construct(path):
    static = {}
    with open(path) as fi:
        for line in fi:
            _, count = line.strip().split('\t')
            try:
                count = int(count)
            except:
                continue
            num = static.setdefault(count, 0)
            static[count] = num + 1
        return static

def get_percent(static):
    total = sum(static.values())
    for count in static.keys():
        static[count] /= total
    return static

def get_cdf(static):
    cdf = {}
    percent_cul = 0.0
    for count in sorted(static.keys()):
        cdf[count] = static[count] + percent_cul
        percent_cul = cdf[count]
    return cdf

def main():
    path = sys.argv[1]
    static = construct(path)
    with open(sys.argv[2], "w") as w_fi:
        for key in sorted(static.keys()):
            w_fi.write("{0}:{1}\n".format(key, static[key]))

if __name__ == "__main__":
    main()
