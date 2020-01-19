import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv1D, Flatten, MaxPooling1D
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
########################### reading and spliting data #########################
np.random.seed(80)

xrp = np.load('xrp.npy')
'''for i in range(xrp.shape[1]):
    plt.figure(i)
    plt.plot(xrp[:,i])'''

# normalize the dataset, LSTM is very sensetive to noise
'''scaler = MinMaxScaler(feature_range=(0, 1))
xrp = scaler.fit_transform(xrp)'''
#m = [4,8,16,32,64,128]
#for i in m:
for i in range(8):
#print(i)
    x = xrp[:,i]
    
    lookback = 144 #lookback for 12 hours
    delay = 36 #number of readings in 3 hours
    pred_win = 12 #prediction for next hour
    
    rows = x.shape[0]-lookback-delay-pred_win+1
    col = lookback
    
    X =  np.zeros((rows,col))
    Y =  np.zeros((rows,))
    pred =  np.zeros((rows,))
    
    start = time.time() 
    
    for i in range(rows):
        X[i] = (x[i:lookback+i])
        Y[i] = (np.mean(x[i+lookback+delay:i+lookback+delay+pred_win]))
        
    #splitting to train an test
    
    '''x_train = X[0:1300000,:]
    y_train = Y[0:1300000,]
    x_test = X[1300000:,:]
    y_test = Y[1300000:,]'''
    
    '''x_train = X[0:130000,:]
    y_train = Y[0:130000]
    x_test = X[130000:200000,:]
    y_test = Y[130000:200000] #2 not work well '''
    
    x_train = X[0:50000,:]
    y_train = Y[0:50000,]
    x_test = X[50000:70000,:]
    y_test = Y[50000:70000,]
    #print()
    #np.save('xRay_X', X)
    #np.save('xRay_Y',Y)
    #np.save('xRay_pred',pred)
    ################################ baseline #####################################
    '''start = time.time()  
    #train_pred = np.mean(x_train , axis=1)
    #print('Train MSE  {0:.4f} '.format(np.mean(np.square(train_pred-y_train)))) 
    test_pred = np.mean(x_test , axis=1)
    elapsed_time = time.time()-start
    print('{0:.4f} '.format(elapsed_time))#Running Time 
    print('{0:.4f} '.format(np.mean(np.square(test_pred-y_test))))#Test MSE
    print()'''
    ################################ LSTM #########################################
    #n=[2,4,8,16,32,64]
    #b=[32,564,128]
    #for i in b:
    # create and fit the LSTM network
    batch_size = 100#128#1,2,4 number of samples in test=476 which is divisible by 1,2,4, for stateful
    epochs = 20
    
    # reshape input to be [samples, time steps, features], 
    #time steps=1:one-time step for each sample
    x_train_ls = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    y_train_ls = np.reshape(y_train, (y_train.shape[0], 1, 1))
    x_test_ls = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    y_test_ls = np.reshape(y_test, (y_test.shape[0], 1, 1))
    
    model = Sequential()
    #LSTM: activation='tanh'
    model.add(LSTM(64,input_shape=x_train_ls.shape[1:],return_sequences =True
                   ,stateful = True,batch_size=batch_size))
    #model.add(BatchNormalization())#))#,
    model.add(LSTM(128,return_sequences =True,stateful = True,batch_size=batch_size))#,stateful = True,batch_size=batch_size
    #,batch_size=batch_size, stateful=True
    #https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
    #batch_size=batch_size, stateful=True
    #return_sequences =True : required configuration for using LSTM
    #Dense: activation=None
    model.add(Dense(1,activation='linear'))
    
    #model.summary()
    
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    
    start = time.time()
    elapsed_time = time.time()-start
    
    history = model.fit(x_train_ls, y_train_ls,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,#animated progress bar
                    validation_data=(x_test_ls, y_test_ls))
    
    elapsed_time = time.time()-start
    score = model.evaluate(x_test_ls, y_test_ls, verbose=0,batch_size=batch_size)#,batch_size=batch_size
    print('{0:.4f} '.format(elapsed_time)) 
    #print('{0:.4f}'.format(score[0]))#Test loss:
    print('{0:.4f}'.format(score[1]))#Test mean_squared_error:
    print()
    ################################ CONV1D #######################################
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    x_train_1D = x_train.reshape((x_train.shape[0],x_train.shape[1],n_features))
    x_test_1D = x_test.reshape((x_test.shape[0],x_test.shape[1],n_features))
    y_test_1D = y_test.reshape((y_test.shape[0],n_features))
    y_train_1D = y_train.reshape((y_train.shape[0],n_features))
    
    batch_size = 32#1,2,4 number of samples in test=476 which is divisible by 1,2,4, for stateful
    epochs = 1
    
    model = Sequential()
    #conv1D: activation=None
    model.add(Conv1D(32, 3,input_shape=x_train_1D.shape[1:],activation='relu'))
    model.add(Conv1D(32, 3,activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))
    #model.add(Dropout(0.25))
    
    
    #model.add(Conv1D(8,3, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=(2)))
    #model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1,activation='linear'))
    
    #model.summary()
    
    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['mse'])
    
    start = time.time()
    elapsed_time = time.time()-start
    history = model.fit(x_train_1D, y_train_1D,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,#animated progress bar
                    validation_data=(x_test_1D, y_test_1D))
    
    elapsed_time = time.time()-start
    score = model.evaluate(x_test_1D, y_test_1D, verbose=0)
    
    print('{0:.4f} '.format(elapsed_time)) 
    #print('{0:.4f}'.format(score[0]))#Test loss:
    print('{0:.4f}'.format(score[1]))#Test mean_squared_error:
    print()
    ############################# regression tree #################################
    #model = DecisionTreeRegressor(random_state=80)
    model = RandomForestRegressor(random_state=80, n_estimators=100)
    start = time.time()
    model.fit(x_train,y_train)
    elapsed_time = time.time()-start
    pred = model.predict(x_test)
    print('{0:.4f} '.format(elapsed_time)) 
    print('{0:.4f}'.format(np.mean(np.square(pred-y_test))))
    print("\n","\n")
    
#Baseline	﻿
a= [0.1576, 0.0449, 0.2863, 0.0183, 0.0115, 0.0110, 0.0130, 0.0196]
#LSTM 	
b= [0.1553, 0.0475, 0.1609, 0.0210, 0.0137, 0.0214, 0.0178, 0.0176]
#CONV1D 
c= [0.1527, 0.0562, 0.2009, 0.0163, 0.0149, 0.0170, 0.0156, 0.0170]
#RandomForest 
d= [	0.1976, 0.0465, 0.1606, 0.0214, 0.0170, 0.0184, 0.0147, 0.0145]
#cols
y=[0,1,2,3,4,5,6,7]
import numpy as np
#import matplotlib.pyplot as plt

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(y,a, 'k-.', label='Baseline')
ax.plot(y,b, 'k:', label='LSTM')
ax.plot(y,c, 'k--', label='CONV1D')
ax.plot(y, d,'k', label='RandomForest')

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()

#Runing Time
#Baseline	
a=[0.0340,0.0029,0.0051,0.0033,0.0026,0.0027,0.0026,0.0035 ] 
#LSTM,	
b=[126.1001 ,112.1338 ,108.2297 ,109.3996 ,107.487 , 113.1581 ,108.8131	,118.435]
#CONV1D	
c = [13.8649	, 14.9055	,15.4216	, 20.2903, 18.3685	,14.9746	, 17.4473 ,18.5371]
#RandomForest
d=	[941.1236 ,932.5938 ,885.8356 ,1355.2813 ,1186.3088 ,984.7476 ,909.8466 ,802.5940]

fig, ax = plt.subplots()
ax.plot(y,a, 'k-.', label='Baseline')
ax.plot(y,b, 'k:', label='LSTM')
ax.plot(y,c, 'k--', label='CONV1D')
ax.plot(y, d,'k', label='RandomForest')

legend = ax.legend(loc='center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()
