import pandas as pd


def read_time(path):
    r=pd.read_csv(path,sep=";")
    onset=[]
    offset=[]
    for i in range(len(r.index)):
        d=r.iloc[i,0]
        pos1=d.find('\t')
        pos2 = d.rfind('\t')
        onset.append(eval(d[pos1+1:pos2]))
        offset.append(eval(d[pos2 + 1:]))
    return onset,offset