import sys
sys.path.append('/home/zhouyue/WorkSpace')#在Ubuntu下运行

nb_epoch=500
nb_epoch_cont=100
batch_size=32
T=48
lr=0.0002#lr=0.0002
len_closeness=3
len_period=1
len_trend=1
nb_residual_unit=12
nb_flow=2
day_test=7*4
len_test=T*day_test
map_height,map_width=32,32

model_path='/home/zhouyue/WorkSpace/MyDeepST/Checkpoint_dir'
cache_path='/home/zhouyue/WorkSpace/MyDeepST/cache_data'
result_path='/home/zhouyue/WorkSpace/MyDeepST/result_dir'