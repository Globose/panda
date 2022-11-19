import pandas as pd
import numpy as np
import math
import os

def entropy(occ):
    sum = 0
    total_occ = occ.sum()
    for o in occ:
        p = o/(total_occ)
        if p != 0:
            sum -= p*math.log2(p)
    return sum

def gain(lst):
    sum = lst[0][1]
    total = lst[0][2]
    for i in range(1,len(lst)):
        sum -= lst[i][1]*lst[i][2]/total
    return sum

def calcEntropy(x, y):
    entropys = []
    yc = y.value_counts()
    entropys.append((x.name, entropy(yc), yc.sum()))
    
    for c in x.cat.categories:
        d = {"yes": 0, "no": 0}
        for index, value in x.items():
            if value == c:
                d[y.get(index)] += 1
        s = pd.Series(d)
        entropys.append((c,entropy(s),s.sum()))
    
    print(entropys)
    print(gain(entropys))

def main():
    df = pd.read_csv("titanic.csv", dtype="category", sep=",", encoding="utf-8")
    y = df.iloc[:,len(df.columns)-1]
    for i in range(len(df.columns)-2):
        x = df.iloc[:,i+1]
        calcEntropy(x,y)
        print()


if __name__ == '__main__':
    main()

    
    