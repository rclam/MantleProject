#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rclam
"""

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow.keras.metrics

from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import sklearn.metrics

from tensorflow.keras.optimizers import Adam


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = np.load("a_T_all.npy")

X = data[:,:-1,:]      # Take only first 20 time step for training
y = data[:,-1,:]       # Target label = last time step

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# standardize feature vecotrs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# alternate scaling option (works fine)
# scalers = {}
# for i in range(X_train.shape[1]):
#     scalers[i] = StandardScaler()
#     X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :]) 

# for i in range(X_test.shape[1]):
#     X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 


nTimeSteps = len(X[0])
iLength = len(y[0])
nSequences = len(y)
print('No. training data (how many sequences): ', nSequences)
print('No. time steps per sequence: ', nTimeSteps)
print('No. data points per time step: ', iLength)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Define LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_shapes = ( nTimeSteps, iLength)

# define LSTM configuration
n_epoch = 1200
n_hidden = 2
f_dropout = 0.2


# build/create LSTM
model = Sequential()
model.add(LSTM(units=n_hidden, 
                return_sequences=True,
                input_shape=input_shapes, activation = 'relu'))
model.add(Dropout(f_dropout))
for s_bool in [True, False]:
    model.add(LSTM(units = 2, return_sequences=s_bool))

# add fully connected output layer
model.add( Dense(units=6912))

# eta = 0.01
# opt = SGD(lr=eta)
# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

opt = Adam(learning_rate=0.2)
model.compile(loss='mean_squared_error', optimizer=opt, 
              metrics=[tensorflow.keras.metrics.Accuracy()])


# check size of ANN
model.summary()
print(model.summary())




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Train LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dResult = model.fit(X_train, y_train, epochs=n_epoch, 
#                      validation_split= 0.2 ,
#                      batch_size=n_batch).history

batch_size=30
dResult = model.fit(X_train, y_train, 
                    batch_size=batch_size, epochs=n_epoch,
                    validation_split=0.2).history

y_pred = model.predict(X_test)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Evaluate LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
R2 = sklearn.metrics.r2_score(y_test, y_pred)
MSE = sklearn.metrics.mean_squared_error(y_test, y_pred)

print('\nR2 score: ', R2)
print('Mean squared Error:', MSE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plt.plot(dResult['loss'],label='loss')
plt.plot(dResult['val_loss'],label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()



plt.plot(dResult['accuracy'],label='train acc.')
plt.plot(dResult['val_accuracy'],label='val acc.')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.show()


# ) Plot histogram of misfit
residual_test = y_pred - y_test
plt.hist(residual_test[3])
plt.ylabel('frequency')
plt.xlabel('misfit for sequence y_{test}[3]')
plt.show()

# plot data distrib. for indiv time step
# plt.subplot(121)
plt.plot(X_test[3,:,:],'o')
plt.xlabel('time step')
plt.ylabel('X value')
plt.xticks(np.arange(0,20+1, 1.0))
plt.show()

# plt.subplot(122)
plt.plot(y_test[3,:],'.')
plt.xlabel('time step')
plt.ylabel('y value')
# plt.xticks(np.arange(0,20+1, 1.0))
plt.tight_layout()
plt.show()


print(model.summary())



