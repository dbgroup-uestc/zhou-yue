import sys
sys.path.append('/home/zhouyue/WorkSpace')#在Ubuntu下运行

import os
import pickle
import time
import numpy as np
import h5py
from MyDeepST.STDeepResnet.ST_Resnet_model import My_model
from MyDeepST.DataProcessing.TaxiBJ_data import load_TaxiBJ_data
import MyDeepST.STDeepResnet.config as cfig

#model_path='/home/zhouyue/WorkSpace/MyDeepST/Checkpoint_dir'
#cache_path='/home/zhouyue/WorkSpace/MyDeepST/cache_data'

if os.path.isdir(cfig.model_path) is False:
    os.mkdir(cfig.model_path)
if os.path.isdir(cfig.cache_path) is False:
    os.mkdir(cfig.cache_path)

def Write_cache(fname,x_train,y_train,x_test,y_test,external_dim, timestamp_train, timestamp_test):
    h5=h5py.File(fname,'w')
    h5.create_dataset('num',data=len(x_train))

    for i, data in enumerate(x_train):#XC,XP,XT,Xmete
        h5.create_dataset('x_train_%i'%i, data=data)
    h5.create_dataset('y_train',data=y_train)
    for i, data in enumerate(x_test):#XC,XP,XT,Xmete
        h5.create_dataset('x_test_%i'%i, data=data)
    h5.create_dataset('y_test',data=y_test)
    h5.create_dataset('external_dim',data=external_dim)
    h5.create_dataset('timestamp_train',data=timestamp_train)
    h5.create_dataset('timestamp_test',data=timestamp_test)

def Read_cache(fname):
    f=h5py.File(fname,'r')
    num=int(f['num'].value)

    x_train,x_test=[],[]
    for i in range(num):
        x_train.append(f['x_train_%i' % i].value)
        x_test.append(f['x_test_%i' % i].value)
    y_test=f['y_test'].value
    y_train=f['y_train'].value
    external_dim=f['external_dim'].value
    timestamp_train=f['timestamp_train'].value
    timestamp_test=f['timestamp_test'].value
    return x_train,y_train,x_test,y_test,external_dim, timestamp_train, timestamp_test

def main():
    ts = time.time()
    #加载数据
    print('loading data')
    fname=os.path.join(cfig.cache_path,'Taxi_data_C{}_P{}_T{}.h5'.format(cfig.len_closeness,cfig.len_period,cfig.len_trend))#后期考虑用config.py
    if os.path.exists(fname):
        x_train, y_train, x_test, y_test, external_dim, timestamp_train, timestamp_test=Read_cache(fname)
        #print('loading X_train shape=:',np.array(x_train).shape)

        print("load %s successfully in cache" % fname)
    else:
        x_train, y_train, x_test, y_test,mmn, external_dim, timestamp_train, timestamp_test=load_TaxiBJ_data(
            len_closeness=cfig.len_closeness,len_period=cfig.len_period,len_trend=cfig.len_trend,
            len_test=cfig.len_test)
        print("load successfully in load_taxi_data")
        Write_cache(fname,x_train, y_train, x_test, y_test, external_dim, timestamp_train, timestamp_test)
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("Training model...")
    My_model(x_train, y_train, x_test, y_test,external_dim, cfig.model_path)



if __name__=='__main__':
    main()
    









