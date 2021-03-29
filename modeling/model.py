# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D

class Model():

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.seq = Sequential()
        self.seq.add(ConvLSTM2D(
            filters=15,
            kernel_size=(3, 9),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation="tanh",
            recurrent_activation="hard_sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            dropout=0.0,
            recurrent_dropout=0.0,
            #**kwargs
        ))
        self.seq.compile(loss='binary_crossentropy', optimizer='adadelta')

    def forward(self, x):
        pass
        # (samples, time, rows, cols, channels)
        # return self.convlstm(x)
