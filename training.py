# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:32:55 2023

@author: fes_map
"""

# system packages
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import time
import winsound as ws
import os
import tensorflow as tf

# customized packages
from Parameter import args
from networks.SpaAG_RAN import SpaAG_RAN

gpu = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "SpaAG-RAN"

# optimizers
rmsprop = RMSprop(learning_rate=0.001)

# here prepare your own 5-d samples
X, X_train, X_test, X_val
y, y_train, y_test, y_val, y_train_1hot, y_test_1hot, y_val_1hot





# produce fake masks
y_sc_train = np.zeros((y_train.shape[0], 1, args.width, args.width, args.band // args.stack))
y_sc_val = np.zeros((y_val.shape[0], 1, args.width, args.width, args.band // args.stack))
y_sc_test = np.zeros((y_test.shape[0], 1, args.width, args.width, args.band // args.stack))

# build models
spaag_ran = SpaAG_RAN(input_shape, args.n_category, args)
model = spaag_ran.model
model.compile(optimizer=rmsprop, loss={"softmax":"categorical_crossentropy", "sc":spaag_ran.spatial_consistency}, 
                    loss_weights=[1., args.sc_weight], metrics={"softmax":"accuracy"})

print(model.summary())

# waiting user to continue
input("model has been built, please input any key to continnue ...\n")





# training
print("training ...\n")
hist = model.fit(x=[X_train], y=[y_train_1hot, y_sc_train], batch_size=args.bs, epochs=args.epochs,
                     verbose=1, validation_data=([X_val], [y_val_1hot, y_sc_val]))    
# play sound
if args.env == 0:
    ws.PlaySound("C:\\Windows\\Media\\Alarm02.wav", ws.SND_ASYNC)
    
# test
print("testing ...\n")
_, _, _, OA = model.evaluate(x, y=[y_test_1hot, y_sc_test])


print("\n", time.ctime(time.time()))

