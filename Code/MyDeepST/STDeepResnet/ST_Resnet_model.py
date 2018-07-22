import sys
sys.path.append('/home/zhouyue/WorkSpace')#在Ubuntu下运行

import collections
import tensorflow as tf

import numpy as np
import MyDeepST.STDeepResnet.config as cfig

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

      Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
          returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
          contains one (depth, depth_bottleneck, stride) tuple for each unit in the
          block to serve as argument to unit_fn.
      """

def subsample(inputs, factor, scope=None):

    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):

    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])  # 填0操作
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        #depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):

            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


#残差单元
@slim.add_arg_scope
def bottleneck(inputs, depth, stride,
               outputs_collections=None, scope=None):

    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        #depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact1 = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact1')  #_BN_Relu
        #if depth == depth_in:  # 该残差单元输入等于输出，就要降采样，stride
        shortcut = subsample(inputs, stride, 'shortcut')
        #else:  # 输入和输出通道不相等，就通过卷积让输入输出通道相等
        #    shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
        #                           normalizer_fn=None, activation_fn=None,
        #                           scope='shortcut')

        #residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
        #                       scope='conv1')
        residual1=slim.conv2d(preact1,depth,[3,3],stride=1,normalizer_fn=None, activation_fn=None,
                             scope='conv1')
        preact2 = slim.batch_norm(residual1, activation_fn=tf.nn.relu, scope='preact2')
        #residual = conv2d_same(residual, depth_bottleneck, 3, stride,
        #                       scope='conv2')
        residual2 = slim.conv2d(preact2, depth, [3, 3], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv2')
        output_maxpool=slim.max_pool2d(residual2,[1,1],stride=1,scope='output_maxpool')
        output = shortcut + output_maxpool

        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                # We do not include batch normalization or activation functions in conv1
                # because the first ResNet unit will perform these. Cf. Appendix of [2].
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

            net=slim.conv2d(net,64,[3,3],activation_fn=None,normalizer_fn=None,padding='SAME')

            net = stack_blocks_dense(net, blocks)
            #res_d = slim.utils.last_dimension(net.get_shape(), min_rank=4)
            #net = slim.conv2d(net, res_d, [1, 1], stride=1, activation_fn=tf.nn.relu, normalizer_fn=None,
            #                  padding='SAME')#激活
            net=tf.nn.relu(net)
            net = slim.conv2d(net, 2, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, padding='SAME')

            #net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')  # 每个块连接时加个BN
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)  # axis=1与axis=2上求平均值
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')  # 全连接层？
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


def st_resnet_v2_12(inputs,num_classes=None,global_pool=None,reuse=None,i=0):#scope='st_resnet_v2_12'
    blocks=[Block('block1',bottleneck,[(64,1)]*12)]
    scope = 'st_resnet_v2_12_%d'%i
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=False, reuse=reuse, scope=scope)

def shuffle_train_data(train_x,train_y):
    train_x_shuffle=[]
    train_x_xc=np.asarray(train_x[0])
    train_x_xp =np.asarray(train_x[1])
    train_x_xt = np.asarray(train_x[2])
    train_x_xmete =np.asarray(train_x[3])
    index=[i for i in range(len(train_y))]
    np.random.shuffle(index)
    train_x_shuffle.append(train_x_xc[index])
    train_x_shuffle.append(train_x_xp[index])
    train_x_shuffle.append(train_x_xt[index])
    train_x_shuffle.append(train_x_xmete[index])
    train_y=train_y[index]
    return train_x_shuffle,train_y


def My_model(train_x,train_y,test_x,test_y,metedata_dim,model_path):
    #超参数，之后考虑在config.py文件中定义
    nb_epoch=cfig.nb_epoch
    nb_epoch_cont=cfig.nb_epoch_cont
    batch_size=cfig.batch_size
    T=cfig.T
    lr=cfig.lr
    len_closeness=cfig.len_closeness
    len_period=cfig.len_period
    len_trend=cfig.len_trend
    nb_residual_unit=cfig.nb_residual_unit
    nb_flow=cfig.nb_flow
    day_test=cfig.day_test
    len_test=cfig.len_test
    map_height,map_width=cfig.map_height,cfig.map_width
    #checkpoint_path='./pretrain/Stresnet.ckpt'

    #输入(占位符) 数据是后面feed进去的
    XC=tf.placeholder(tf.float32,[None,map_height,map_width,nb_flow*len_closeness])
    XP=tf.placeholder(tf.float32,[None,map_height,map_width,nb_flow*len_period])
    XT=tf.placeholder(tf.float32,[None,map_height,map_width,nb_flow*len_trend])
    X_mete=tf.placeholder(tf.float32,[None,metedata_dim])#这里设置成28
    Y=tf.placeholder(tf.float32,[None,map_height,map_width,nb_flow])
    #定义Merge时的变量，变量定义时默认是可以训练的
    weight1=tf.Variable(tf.random_normal([map_height,map_width,nb_flow],stddev=0.35),
                        name='weight1')
    weight2=tf.Variable(tf.random_normal([map_height, map_width, nb_flow],stddev=0.35),
                          name='weight2')
    weight3=tf.Variable(tf.random_normal([map_height, map_width, nb_flow],stddev=0.35),
                          name='weight3')

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        out1,end_points1=st_resnet_v2_12(XC,i=0)#直接要提取的特征，不需要分类
        out2,end_points2=st_resnet_v2_12(XP,i=1)
        out3,end_points3=st_resnet_v2_12(XT,i=2)

    #输出后要把数据merge,out1*w1+out2*w2+out3*w3
    out_merge=out1*weight1+out2*weight2+out3*weight3
    #out_merge = out1 * weight1
    if metedata_dim>0:
        dense1=tf.layers.dense(inputs=X_mete,units=10,activation=tf.nn.relu)
        dense2=tf.layers.dense(inputs=dense1, units=nb_flow*map_width*map_height,
                           activation=tf.nn.relu)
        ex_output=tf.reshape(dense2,[-1,map_height,map_width,nb_flow])
        out=out_merge+ex_output
    else:
        out=out_merge
    main_out=tf.nn.tanh(out)

    #定义loss
    loss=tf.reduce_mean(tf.square(main_out-Y))
    #loss=tf.reduce_sum(tf.pow(main_out-Y,2))/batch_size
    #loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=main_out))
    #定义优化器
    optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    #定义rsem
    rsem=tf.cast(loss**0.5,tf.float32)

    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    saver2=tf.train.Saver(tf.global_variables())
    #model_path='/home/zhouyue/WorkSpace/Checkpoint_dir'

    #print("train_x shape=:",np.array(train_x).shape)

    train_n=int(len(train_x[0]))
    validate_rate=0.1
    validate_len=int(train_n*validate_rate)

    pre_resm2=1000.
    pre_resm1=1000.
    for epoch_i in range(nb_epoch):
        XC_train=train_x[0]
        XP_train=train_x[1]
        XT_train=train_x[2]
        Xmete=train_x[3]
        XC_train_train=XC_train[:-validate_len]
        XC_train_validate=XC_train[-validate_len:]
        XP_train_train = XP_train[:-validate_len]
        XP_train_validate = XP_train[-validate_len:]
        XT_train_train=XT_train[:-validate_len]
        XT_train_validate=XT_train[-validate_len:]
        X_mete_train=Xmete[:-validate_len]
        X_mete_validate=Xmete[-validate_len:]
        Y_train_train=train_y[:-validate_len]
        Y_train_validate=train_y[-validate_len:]
        begin_batch=0
        end_batch=batch_size

        for batch_i in range(int(train_n / batch_size)):
            if end_batch > train_n:
                break
            if train_n - end_batch < batch_size:
                end_batch = train_n - end_batch
            los, _ = sess.run([loss, optimizer], feed_dict={XC: XC_train_train[begin_batch:end_batch],
                                                            XP: XP_train_train[begin_batch:end_batch],
                                                            XT: XT_train_train[begin_batch:end_batch],
                                                            X_mete: X_mete_train[begin_batch:end_batch],
                                                            Y: Y_train_train[begin_batch:end_batch]})
            if batch_i % 20 == 0:
                print('In traing epoch={} Batch_i={} loss={}'.format(epoch_i, batch_i, los))
            begin_batch=end_batch
            end_batch+=batch_size

        #验证
        ls,re=sess.run([loss,rsem],feed_dict={  XC: XC_train_validate,
                                                XP: XP_train_validate,
                                                XT: XT_train_validate,
                                                X_mete: X_mete_validate,
                                                Y: Y_train_validate})

        print('In traing epoch={} validate rsme={} loss={}'.format(epoch_i, re, ls))
        #连续2次resm没变小，就保存模型，并且退出
        #if re>=pre_resm1 and re>=pre_resm2
        #    saver2.save(sess,model_path)
        #    break
        #pre_resm2=pre_resm1
        #pre_resm1=re
        #每个opoch后打乱一次顺序
        train_x,train_y=shuffle_train_data(train_x,train_y)
