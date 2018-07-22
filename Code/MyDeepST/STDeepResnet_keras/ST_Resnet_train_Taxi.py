import sys
sys.path.append('/home/zhouyue/WorkSpace')  # 在Ubuntu下运行

import os
import pickle
import time
import numpy as np
import h5py

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from MyDeepST.STDeepResnet_keras.ST_Resnet_model_keras import stresnet
from MyDeepST.DataProcessing.TaxiBJ_data import load_TaxiBJ_data_ketas
import MyDeepST.STDeepResnet_keras.metrics  as metrics
import MyDeepST.STDeepResnet_keras.config as cfig

# model_path='/home/zhouyue/WorkSpace/MyDeepST/Checkpoint_dir'
# cache_path='/home/zhouyue/WorkSpace/MyDeepST/cache_data'

if os.path.isdir(cfig.model_path) is False:
    os.mkdir(cfig.model_path)
if os.path.isdir(cfig.cache_path) is False:
    os.mkdir(cfig.cache_path)
if os.path.isdir(cfig.result_path) is False:
    os.mkdir(cfig.result_path)


def Write_cache(fname, x_train, y_train, x_test, y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(x_train))

    for i, data in enumerate(x_train):  # XC,XP,XT,Xmete
        h5.create_dataset('x_train_%i' % i, data=data)
    h5.create_dataset('y_train', data=y_train)
    for i, data in enumerate(x_test):  # XC,XP,XT,Xmete
        h5.create_dataset('x_test_%i' % i, data=data)
    h5.create_dataset('y_test', data=y_test)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('timestamp_train', data=timestamp_train)
    h5.create_dataset('timestamp_test', data=timestamp_test)


def Read_cache(fname):
    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    mmn = pickle.load(open(cfig.preprocess_name, 'rb'))
    x_train, x_test = [], []
    for i in range(num):
        x_train.append(f['x_train_%i' % i].value)
        x_test.append(f['x_test_%i' % i].value)
    y_test = f['y_test'].value
    y_train = f['y_train'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['timestamp_train'].value
    timestamp_test = f['timestamp_test'].value
    return x_train, y_train, x_test, y_test,mmn, external_dim, timestamp_train, timestamp_test

def build_model(external_dim):
    c_conf = (cfig.len_closeness, cfig.nb_flow, cfig.map_height,
              cfig.map_width) if cfig.len_closeness > 0 else None
    p_conf = (cfig.len_period, cfig.nb_flow, cfig.map_height,
              cfig.map_width) if cfig.len_period > 0 else None
    t_conf = (cfig.len_trend, cfig.nb_flow, cfig.map_height,
              cfig.map_width) if cfig.len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=cfig.nb_residual_unit)#得到模型
    adam = Adam(lr=cfig.lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()#模型概况
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model

def main():
    print("loading data...")
    ts = time.time()
    fname = os.path.join(cfig.cache_path, 'TaxiBJ_C{}_P{}_T{}_External_kerasdata.h5'.format(
        cfig.len_closeness,cfig.len_period,cfig.len_trend))

    if os.path.exists(fname):
        X_train,Y_train,X_test,Y_test,mmn,external_dim,timestamp_train,timestamp_test=Read_cache(fname)
        print("load %s successfully" % fname)
    else:
        X_train,Y_train,X_test,Y_test,mmn,external_dim,timestamp_train,timestamp_test = load_TaxiBJ_data_ketas(
            T=cfig.T,preprocess_name=cfig.preprocess_name,nb_flow=cfig.nb_flow, len_closeness=cfig.len_closeness, len_period=cfig.len_period,len_trend=cfig.len_trend,
            len_test=cfig.len_test,mete_data=True,meteorol_data=False, holiday_data=False)
        Write_cache(fname,X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::cfig.T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    ts = time.time()
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}.withExternal_keras'.format(
        cfig.len_closeness, cfig.len_period, cfig.len_trend, cfig.nb_residual_unit, cfig.lr)
    fname_param = os.path.join(cfig.model_path, '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print("\nelapsed time (compiling model): %.3f seconds\n" %
          (time.time() - ts))  # 编译模型用时

    print('=' * 10)
    print("training model...")  # 这里训练模型
    ts = time.time()

    history = model.fit(X_train, Y_train,
                        nb_epoch=cfig.nb_epoch,  # 在500次中最好的那次就停下
                        batch_size=cfig.batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)  # 训练

    model.save_weights(os.path.join(
        cfig.model_path, '{}.h5'.format(hyperparams_name)), overwrite=True)  # 保存权重值W

    pickle.dump((history.history), open(os.path.join(
        cfig.result_path,'{}.history.pkl'.format(hyperparams_name)), 'wb'))  # 保存模型

    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')  # 验证模型
    ts = time.time()
    model.load_weights(fname_param)  # 使用之前保存好的权重值
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))#**************
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))#*****************
    print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("training model (cont)...")
    ts = time.time()
    fname_param = os.path.join(
        cfig.model_path, '{}.cont.best.h5'.format(hyperparams_name))
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')

    history = model.fit(X_train, Y_train, nb_epoch=cfig.nb_epoch_cont, verbose=2, batch_size=cfig.batch_size, callbacks=[
        model_checkpoint])  # 这里没有Early stop，是按照指定的步数来训练
    pickle.dump((history.history), open(os.path.join(
        cfig.result_path, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    model.save_weights(os.path.join(
        cfig.model_path, '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the final model')  # 这里再次验证模型，与之前有Early stop的模型进行比较
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                                                            0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    ts = time.time()
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
    print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))


if __name__ == '__main__':
    main()










