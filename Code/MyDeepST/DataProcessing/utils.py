import sys
sys.path.append('/home/zhouyue/WorkSpace')#在Ubuntu下运行

import h5py
import time
import pandas as pd
from datetime import datetime,timedelta

def load_stdata(fname):
    f=h5py.File(fname)
    data=f['data'].value
    timeslots=f['date'].value
    f.close()
    return data,timeslots

#返回的是一段时间戳对应的pd.Timestamp列表
def string2timestamp(strings, T=48):
    timestamps=[]
    time_per_slot=24.0/T
    num_per_T=T//24
    for s in strings:
        year,month,day,slot =int(s[:4]),int(s[4:6]),int(s[6:8]),int(s[8:])-1
        #print("year=:",year,'month=:',month,"day=:",day,'slot=:',slot)
        Time=datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))
        #print('Time=',Time)
        timestamps.append(pd.Timestamp(Time))
    return timestamps

#移除不完整的天数
def remove_incomplete_days(data,timeslots,T):
    days=[]
    incomplete_days=[]

    i=0
    while i<len(timeslots):
        if int(timeslots[i][8:]) != 1:
            i+=1
            continue
        elif (i+T-1)<len(timeslots) and int(timeslots[i+T-1][8:])==T:
            days.append(timeslots[i][:8])
            i+=1
        else:
            incomplete_days.append(timeslots[i][:8])
            i+=1
    print('incomplete_days: ',incomplete_days)

    idx=[]
    days=set(days)
    for i,slot in enumerate(timeslots):
        if slot[:8] in days:
            idx.append(i)

    data=data[idx]
    timeslots=timeslots[idx]
    return data,timeslots

def timestamp2vec(timestamps):
    ret=[]
    #timestamps2=list
    #for t in timestamps:
    #    timestamps2=t
    #vec=[time.strptime(t[:8],'%Y%m%d').tm_wday for t in timestamps]
    vec=[]
    for t in timestamps:
        #print(str(t[:8],encoding='UTF-8'),type(str(t[:8],encoding='UTF-8')))
        dt=datetime.strptime(str(t[:8],encoding='UTF-8'),'%Y%m%d')
        vec.append(dt.weekday())

    #print('vec len',len(vec))
    #print('\n')
    #print(vec)
    for i in vec:
        v=[0 for _ in range(7)]
        #print('i=:',i)
        v[i]=1
        if i>=5:#weekends
            v.append(1)
        else:
            v.append(0)
        ret.append(v)
    return ret