
from MyDeepST.DataProcessing.utils import string2timestamp
import os
import numpy as np
import pandas as pd

class STMatrix(object):
    def __init__(self,data,Timestamps,T=48,CheckComplete=True):
        self.data=data
        self.Timestamps=Timestamps
        #print('1:',len(self.Timestamps))
        self.T=T
        self.pd_Timestamps=string2timestamp(Timestamps,T)#****
        #print(self.pd_Timestamps)
        #print("2:",len(self.pd_Timestamps))
        self.Timestamp_index=[]
        self.Timestampi_index=[]

        if CheckComplete:
            self.CheckComplete()
        self.make_index()

    # def make_index(self):  #时间戳字典
    #     self.get_index=dict()
    #     print('1:',len(self.pd_Timestamps))
    #     for i,ts in enumerate(self.pd_Timestamps):
    #         self.get_index[ts]=i
    #     print('2:', len(self.get_index.keys()))

    def Get_index(self,i):
        return self.get_matrix(self.Timestamp_index[i])

    #字典一直无法存储啊，改用列表了
    def make_index(self):
        for i,ts in enumerate(self.pd_Timestamps):
            self.Timestamp_index.append(ts)
            self.Timestampi_index.append(i)
        #print(len(self.Timestamp_index),len(self.Timestampi_index))



    #检查pd_Timestamp数据集的完整性
    def CheckComplete(self):
        Missing_timestamp=[]
        offset=pd.DateOffset(minutes=24 * 60 // self.T)
        i=1
        pd_Timestamps=self.pd_Timestamps
        while i < len(pd_Timestamps):
            if pd_Timestamps[i-1]+offset != pd_Timestamps[i+1]:
                Missing_timestamp.append("(%s--%s)"%(pd_Timestamps[i-1],pd_Timestamps[i]))
            i+=1
        for v in Missing_timestamp:
            print(v)
        assert len(Missing_timestamp)==0

    #根据时间戳获取对应数据Tensor
    # def get_matrix(self,timestampe):
    #     i=0
    #     for i in range(len(self.pd_Timestamps)):
    #         if self.pd_Timestamps[i] == timestampe:
    #             break
    #     return self.data[i]

    #直接根据整数索引来返回Tensor，没办法，get_index用不了，不能根据时间戳来返回
    def get_matrix(self,i):
        return self.data[i]

    #检查depend中的数据是否在字典中，以防查不到
    def check_it(self, depends):
        #print(len(self.get_index.keys()))
        #print('dependes: ',depends)
        #print(depends)
        #print(self.Timestamp_index)
        for d in depends:
            #print(d)
            if d not in self.Timestamp_index:#pd_timestamp
                return False
        return True

    #创建数据集
    def create_dataset(self,len_clossness=3,len_trend=3,TrendInterver=7,len_period=3,PeriodInterver=1):
        #assert (len_clossness+len_period+len_period)>1
        offset=pd.DateOffset(minutes=24 * 60 // self.T)
        XC=[]
        XP=[]
        XT=[]
        Y=[]
        timestamps_Y = []
        Test_I=[]
        #print(self.data.shape,self.Timestamps.shape,np.array(self.pd_Timestamps).shape)

        dependes=[range(1,len_clossness+1),[PeriodInterver*self.T*j for j in range(1,len_period+1)]
                  ,[TrendInterver*self.T*j for j in range(1,len_trend+1)]]
        i=max(self.T*PeriodInterver*len_period,self.T*TrendInterver*len_trend,len_clossness)

        #print('Test get_matrix: ',self.get_matrix(self.get_index[1000]))
        while i < len(self.pd_Timestamps):
            Flag=True
            for depende in dependes:
                if Flag is False:
                    break
                Flag=self.check_it([self.pd_Timestamps[i]-j*offset for j in depende])
                #print('Flag=',Flag)

            if Flag is False:
                i+=1
                continue
            #print('*'*10)
            #x_c=[self.get_matrix(self.pd_Timestamps[i]-j*offset) for j in dependes[0]]
            #x_p=[self.get_matrix(self.pd_Timestamps[i]-j*offset) for j in dependes[1]]
            #x_t=[self.get_matrix(self.pd_Timestamps[i]-j*offset) for j in dependes[2]]
            x_c = [self.get_matrix(i-j) for j in dependes[0]]
            x_p = [self.get_matrix(i-j) for j in dependes[1]]
            x_t = [self.get_matrix(i-j) for j in dependes[2]]
            y=self.get_matrix(i)

            if len_clossness>0:
                XC.append(np.vstack(x_c))
            if len_period>0:
                XP.append(np.vstack(x_p))
            if len_trend>0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            Test_I.append(i)
            #timestamps_Y.append(self.Timestamps[i])
            i+=1

        XC=np.asarray(XC)
        XP=np.asarray(XP)
        XT=np.asarray(XT)
        Y=np.asarray(Y)
        for i in Test_I:
            timestamps_Y.append(self.Timestamps[i])#返回的是个列表

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape,
              'timestamps_Y shape',np.array(timestamps_Y).shape)
        #print('timestamp_Y:',timestamps_Y)
        #print(timestamps_Y)
        return XC,XP,XT,Y,timestamps_Y


