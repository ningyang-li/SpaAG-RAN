# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:36:41 2022

@author: fes_map
"""

from tensorflow.keras.layers import Input, Flatten, Dense, Multiply, Dropout, Add
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, Activation, Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .Network import Network


class SpaAG_RAN(Network):
    '''
    paper title:    Spatial attention guided residual attention network for hyperspectral image classification
    input:          5-dimensional tensor
    '''
    def __init__(self, input_shape, n_category, args, scheme=0):
        super().__init__("SpaAG-RAN", input_shape, n_category)
        
        self.args = args        
        self.build_model()
        
    
    def Similarity(self, x):
        '''
        get the similarity between the center spectrum and its neighborhoods
        kernel: L1,  L2,  SAM
        '''
        n_sp, n_channels, n_rows, n_cols, n_bands = x.shape
        # reshape to 4-d tensor
        center = x[:, :, n_rows // 2, n_cols // 2, :]
        center = K.reshape(center, (-1, 1, n_bands))
        
        x_2d = K.reshape(x, (-1, n_rows * n_cols, n_bands))
        # expand to a matrix with the same size of x_2d
        center = K.tile(center, [1, n_rows * n_cols, 1])
        
        # compute the L2-similarity
        sim = K.sum(K.pow(x_2d - center, 2), axis=-1)
        sim = sim / (K.max(sim, axis=-1, keepdims=True) + K.epsilon())
        
        sim = K.reshape(sim, (-1, 1, n_rows, n_cols, 1))
        
        return sim
    
    
    def ISS_Sigmoid(self, sim):
        mask = 1. / (1. + K.exp(self.args.alpha * (sim - self.args.t)))
    
        return mask


    def f_Reshape(self, x, shape):
        return K.reshape(x, shape)
    

    def SpaAM(self, x, nid=0):
        '''
        spatial atention module
        '''
        n_channels, n_rows, n_cols, n_bands = x.shape[1:]
        # channel=1
        xp = Conv3D(filters=1, kernel_size=1, strides=1, data_format=self.DT)(x)
        # simlarity
        sim = Lambda(self.Similarity, output_shape=(1, n_rows, n_cols, 1), name="similarity" + str(nid))(xp)
        # mask
        mask = Lambda(self.ISS_Sigmoid, output_shape=(1, n_rows, n_cols, 1), name="ISS_Sigmoid" + str(nid))(sim)
        
        return mask
    
    
    def SpeAM(self, x):
        '''
        spectral attention module
        '''
        n_channels, n_rows, n_cols, n_bands = x.shape[1:]
        # channel=1
        xp = Conv3D(filters=1, kernel_size=1, strides=1, activation="relu", data_format=self.DT)(x)
        # maxpooling across the spatial dimensional
        xp = AveragePooling3D(pool_size=(n_rows, n_cols, 1), strides=(n_rows, n_cols, 1), data_format=self.DT)(xp)
        # MLP
        fxp = Flatten()(xp)
        fxp = Dense(n_bands // 2, activation="relu")(fxp)
        fxp = Dense(n_bands, activation="sigmoid")(fxp)
        
        mask = Lambda(self.f_Reshape, output_shape=(1, 1, 1, n_bands), arguments={"shape":(-1, 1, 1, 1, n_bands)})(fxp)
        
        return mask
    
    
    def Res_Block(self, x, n_filters_output, kernel_size):
        n_channels, n_rows, n_cols, n_bands = x.shape[1:]
        
        if n_channels != n_filters_output:
            # shortcut
            sc = Conv3D(filters=n_filters_output, kernel_size=1, strides=1, data_format=self.DT)(x)
        else:
            sc = x
        
        xp = Conv3D(filters=n_filters_output, kernel_size=kernel_size, strides=1, padding="same", data_format=self.DT)(x)
        # xp=BatchNormalization()(xp)
        xp = Activation("relu")(xp)
        xp = Conv3D(filters=n_filters_output, kernel_size=kernel_size, strides=1, padding="same", data_format=self.DT)(xp)
        
        # Add
        xp = Add()([xp, sc])
        # xp = BatchNormalization()(xp)
        xp = Activation("relu")(xp)
        
        return xp
    
    
    def SSFEM(self, x, filters, kernel_size):
        '''
        stack   depth
        e.g.:   filters=[8, 16, 32]
                ks=kernel_size=[[1, 1, 3], [1, 1, 5], [1, 1, 7]]
        '''
        n_channels, n_rows, n_cols, n_bands = x.shape[1:]
        
        s = self.Res_Block(x, filters[0], kernel_size[0])
        
        for i in range(1, self.args.stack):
            s = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), data_format=self.DT)(s)
            s = self.Res_Block(s, filters[i], kernel_size[0])
        
        s = AveragePooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), data_format=self.DT, name="feature")(s)
        
        return s
    
    
    def build_model(self):
        # Input
        x = Input(shape=self.input_shape)
        # SpaAM
        ma1 = self.SpaAM(x, nid=0)
        # SpeAM
        xa = Multiply()([x, ma1])
        me = self.SpeAM(xa)
        # SSFEM
        xe = Multiply()([x, me])
        xp = self.SSFEM(xe, self.args.filters, self.args.kernel_size)
        # SpaAM
        ma2 = self.SpaAM(xp, nid=1)
        # spatial_consistency
        ma = Concatenate(name="sc")([ma1, ma2])
        xp = Multiply()([xp, ma2])
        # FC
        F = Flatten()(xp)
        F = Dropout(0.3)(F)
        P = Dense(self.n_category, activation="softmax", name="softmax")(F)
    
        # build model
        self.model = Model(inputs=x, outputs=[P, ma], name="SpaAG-RAN")

        print(self.name + " build success")
        
    
    def spatial_consistency(self, y_true, y_pred):
        mae = K.sum(K.abs(y_pred[0] - y_pred[1])) / K.constant(self.args.bs, "float32")
        
        return mae
    

