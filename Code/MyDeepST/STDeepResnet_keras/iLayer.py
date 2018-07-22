from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import numpy as np


class iLayer(Layer):  #继承于Layer类
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])#第一个是batch，所以不用训练
        self.W = K.variable(initial_weight_value)#创建变量，同时定义为类中变量
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape
