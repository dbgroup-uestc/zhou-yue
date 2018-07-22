import sys
sys.path.append('/home/zhouyue/WorkSpace')#在Ubuntu下运行

from MyDeepST.DataProcessing.utils import *
from MyDeepST.DataProcessing.STMatrix import STMatrix
from MyDeepST.DataProcessing.Minmax_normalization import MinMaxNormalization

import numpy as np
import h5py
import copy
import pickle

def load_holiday(timeslots,fname='/home/zhouyue/WorkSpace/MyDeepST/DataProcessing/TaxiBJ/BJ_Holiday.txt'):
    f=open(fname,'r')
    holidays=f.readlines()
    holidays=[h.strip() for h in holidays]
    #print('timeslots=: ',timeslots)#传进来的是二维列表
    #timeslots_2=list
    #for t in timeslots:
    #    timeslots_2=t
    H=np.zeros(len(timeslots))
    for i,slot in enumerate(timeslots):
        if slot in holidays:
            H[i]=1
    #print('H.sum()=:',H.sum())
    #print(H[:,None])
    return H[:None]

def load_meteorol(timeslots, fname='/home/zhouyue/WorkSpace/MyDeepST/DataProcessing/TaxiBJ/BJ_Meteorology.h5'):
    f=h5py.File(fname,'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M=dict()
    for i,slot in enumerate(Timeslot):
        M[slot]=i
    #print('Len M=: ',len(M))

    WS=[]
    WR=[]
    TE=[]
    #timeslots_2=list
    #for s in timeslots:
    #    timeslots_2=s

    #print('timeslots=:',timeslots)
    for slot in timeslots:
        #print(slot)
        predict_id=M[slot]
        cur_id=predict_id-1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    #全是timeslot前一个时刻的风速，天气，温度
    WS=np.asarray(WS)
    WR=np.asarray(WR)
    TE=np.asarray(TE)
    #print("WS: ",WS,'WR: ',WR,"TE: ",TE)
    WS=1.*(WS-WS.min())/(WS .max()-WS.min())
    TE=1.*(TE-TE.min())/(TE.max()-TE.min())
    #print("shape: ", WS.shape, WR.shape, TE.shape)
    merge_meta=np.hstack([WR,WS[:,None],TE[:,None]])
    #print(WR.shape,WS[:,None].shape,TE[:,None])
    #print(merge_meta.shape)
    return merge_meta

def load_TaxiBJ_data(T=48,len_closeness=None,len_period=None,len_trend=None,nb_flow=2,len_test=None
                     ,mete_data=True,meteorol_data=True, holiday_data=True):
    assert (len_closeness+len_period+len_trend)>0
    data_all=[]
    data_all1=[]
    timestamp_all=[]

    for year in range(13,17):
        fname='/home/zhouyue/WorkSpace/MyDeepST/DataProcessing/TaxiBJ/BJ{}_M32x32_T30_InOut.h5'.format(year)
        data,timeslots=load_stdata(fname)
        data,timeslots=remove_incomplete_days(data,timeslots,T)#移除了不完整天数对应的data,timeslot
        data=data[:,:nb_flow]
        data[data<0]=0
        data_all.append(data)
        data_all1.append(data)
        timestamp_all.append(timeslots)

    data_train=np.vstack(data_all1)[:-len_test]
    print('data_train.shape: ',data_train.shape)
    mmn=MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn=[mmn.Transfor(d) for d in data_all]

    #pickle.dump()??????

    XC=[]
    XP=[]
    XT=[]
    Y=[]
    Timestamp_Y=[]
    for data,timestamp in zip(data_all,timestamp_all):
        #print(data.shape,timestamp.shape)
        st=STMatrix(data,timestamp,T,CheckComplete=False)
        _XC,_XP,_XT,_Y,_timestamps_Y=st.create_dataset(
            len_clossness=len_closeness,len_period=len_period,len_trend=len_trend)
        #print(np.array(_XC).shape,np.array(_XP).shape,np.array(_XT).shape,np.array(_timestamps_Y).shape)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        Timestamp_Y.extend(_timestamps_Y)#**************
    #print('Time_stamp_Y=: ',Timestamp_Y)

    mete_feature=[]
    if mete_data:
        time_feature=timestamp2vec(Timestamp_Y)#Timestamp_Y是预测人流对应的时间
        mete_feature.append(time_feature)
        print("time_feature shape=: ",np.array(time_feature).shape)
    if holiday_data:
        holiday_feature=load_holiday(Timestamp_Y)
        holiday_feature_1=np.array(holiday_feature).reshape(len(holiday_feature),-1)
        mete_feature.append(holiday_feature_1)
        print("holiday_feature shape=: ", holiday_feature_1.shape)
    if meteorol_data:
        meteorol_feature=load_meteorol(Timestamp_Y)
        mete_feature.append(meteorol_feature)
        print("meteorol_feature shape=: ", np.array(meteorol_feature).shape)

    #print("mete_feature shape=: ", np.array(mete_feature).shape)

    mete_feature = np.hstack(mete_feature) if len(mete_feature)>0 else \
        np.asarray(mete_feature)
    print("mete_feature shape=: ", np.array(mete_feature).shape)

    metedata_dim = mete_feature.shape[1] if len(mete_feature.shape)>0 else None

    #if mete_data and meteorol_data and holiday_data:
        #print('Time_feature: ',time_feature.shape,'holiday_feature: ',holiday_feature.shape,
        #      'meteorol_feature: ',meteorol_feature.shape)

    XC=np.vstack(XC)
    XP=np.vstack(XP)
    XT=np.vstack(XT)
    Y=np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = Timestamp_Y[:-len_test], Timestamp_Y[-len_test:]

    X_train=[]
    X_test=[]

    for l,X_ in zip([len_closeness,len_period,len_trend],[XC_train,XP_train,XT_train]):
        if l >0:
            X_train.append(X_)
    for l,X_ in zip([len_closeness,len_period,len_trend],[XC_test,XP_test,XT_test]):
        if l >0:
            X_test.append(X_)

    #print('X_train: ',X_train.shape,'Y_train: ',Y_train.shape,
    #      'X_test: ',X_test.shape,'Y_test',Y_test.shape)

    if metedata_dim is not None:
       metedata_train,metedata_test=mete_feature[:-len_test],mete_feature[-len_test:]
       X_train.append(metedata_train)
       X_test.append(metedata_test)

    X_train_reshape=[]
    Y_train_reshape=[]
    X_test_reshape=[]
    Y_test_reshape=[]
    i=0
    for _X in X_train:
        if i>2:
            break
        #print('before X_train',_X.shape)
        _X=np.transpose(_X,(0,2,3,1))
        i+=1
        X_train_reshape.append(_X)
        #print('after X_train', _X.shape)
    X_train_reshape.append(X_train[3])
    #print()
    Y_train=np.transpose(Y_train,(0,2,3,1))
    #print()

    for j in range(3):
        #print('before X_test', X_test[j].shape)
        X_test[j]=np.transpose(X_test[j], (0, 2, 3, 1))
        X_test_reshape.append(X_test[j])
        #print('after X_test', X_test[j].shape)
    X_test_reshape.append(X_test[3])
    Y_test=np.transpose(Y_test,(0,2,3,1))
    #print()
    return X_train_reshape, Y_train, X_test_reshape, Y_test, mmn, metedata_dim, timestamp_train, timestamp_test


def load_TaxiBJ_data_ketas(T=48,len_closeness=None,len_period=None,len_trend=None,nb_flow=2,
                           preprocess_name='preprocessing.pkl',len_test=None
                     ,mete_data=True,meteorol_data=True, holiday_data=True):
    assert (len_closeness+len_period+len_trend)>0
    data_all=[]
    data_all1=[]
    timestamp_all=[]

    for year in range(13,17):
        fname='/home/zhouyue/WorkSpace/MyDeepST/DataProcessing/TaxiBJ/BJ{}_M32x32_T30_InOut.h5'.format(year)
        data,timeslots=load_stdata(fname)
        data,timeslots=remove_incomplete_days(data,timeslots,T)#移除了不完整天数对应的data,timeslot
        data=data[:,:nb_flow]
        data[data<0]=0
        data_all.append(data)
        data_all1.append(data)
        timestamp_all.append(timeslots)

    data_train=np.vstack(data_all1)[:-len_test]
    print('data_train.shape: ',data_train.shape)
    mmn=MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn=[mmn.Transfor(d) for d in data_all]

    #pickle.dump()??????
    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC=[]
    XP=[]
    XT=[]
    Y=[]
    Timestamp_Y=[]
    for data,timestamp in zip(data_all,timestamp_all):
        #print(data.shape,timestamp.shape)
        st=STMatrix(data,timestamp,T,CheckComplete=False)
        _XC,_XP,_XT,_Y,_timestamps_Y=st.create_dataset(
            len_clossness=len_closeness,len_period=len_period,len_trend=len_trend)
        #print(np.array(_XC).shape,np.array(_XP).shape,np.array(_XT).shape,np.array(_timestamps_Y).shape)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        Timestamp_Y.extend(_timestamps_Y)#**************
    #print('Time_stamp_Y=: ',Timestamp_Y)

    mete_feature=[]
    if mete_data:
        time_feature=timestamp2vec(Timestamp_Y)#Timestamp_Y是预测人流对应的时间
        mete_feature.append(time_feature)
        print("time_feature shape=: ",np.array(time_feature).shape)
    if holiday_data:
        holiday_feature=load_holiday(Timestamp_Y)
        holiday_feature_1=np.array(holiday_feature).reshape(len(holiday_feature),-1)
        mete_feature.append(holiday_feature_1)
        print("holiday_feature shape=: ", holiday_feature_1.shape)
    if meteorol_data:
        meteorol_feature=load_meteorol(Timestamp_Y)
        mete_feature.append(meteorol_feature)
        print("meteorol_feature shape=: ", np.array(meteorol_feature).shape)

    #print("mete_feature shape=: ", np.array(mete_feature).shape)

    mete_feature = np.hstack(mete_feature) if len(mete_feature)>0 else \
        np.asarray(mete_feature)
    print("mete_feature shape=: ", np.array(mete_feature).shape)

    metedata_dim = mete_feature.shape[1] if len(mete_feature.shape)>0 else None

    #if mete_data and meteorol_data and holiday_data:
        #print('Time_feature: ',time_feature.shape,'holiday_feature: ',holiday_feature.shape,
        #      'meteorol_feature: ',meteorol_feature.shape)

    XC=np.vstack(XC)
    XP=np.vstack(XP)
    XT=np.vstack(XT)
    Y=np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = Timestamp_Y[:-len_test], Timestamp_Y[-len_test:]

    X_train=[]
    X_test=[]

    for l,X_ in zip([len_closeness,len_period,len_trend],[XC_train,XP_train,XT_train]):
        if l >0:
            X_train.append(X_)
    for l,X_ in zip([len_closeness,len_period,len_trend],[XC_test,XP_test,XT_test]):
        if l >0:
            X_test.append(X_)

    #print('X_train: ',X_train.shape,'Y_train: ',Y_train.shape,
    #      'X_test: ',X_test.shape,'Y_test',Y_test.shape)

    if metedata_dim is not None:
       metedata_train,metedata_test=mete_feature[:-len_test],mete_feature[-len_test:]
       X_train.append(metedata_train)
       X_test.append(metedata_test)

    return X_train, Y_train, X_test, Y_test, mmn, metedata_dim, timestamp_train, timestamp_test



if __name__=='__main__':
    pass
    #data,timeslots=load_stdata('./TaxiBJ/BJ13_M32x32_T30_InOut.h5')
    #load_meteorol(timeslots)
    #X_train,Y_train,X_test,Y_test,mmn,metadata_dim,timestamp_train,timestamp_test=\
    #    load_TaxiBJ_data(len_closeness=3,len_period=4,len_trend=4,len_test=500)
    #print('Y_train=: ',Y_train)

    #print("X_train shape",np.array(X_train[0]).shape,'\n')
    #print("X_train shape", np.array(X_train[1]).shape, '\n')
    #print("X_train shape", np.array(X_train[2]).shape, '\n')
    #print("X_train shape", np.array(X_train[3]).shape, '\n')
    #print('length=: ',len(X_train))



