
import numpy as np
np.random.seed(1337)

class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self,data):
        self._min=data.min()
        self._max=data.max()

    def Transfor(self,data):
        data=1.*(data-self._min)/(self._max-self._min)
        data=data*2.-1.
        return data
