import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(1, )))
model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1))
 
model.compile(optimizer=Adam(), loss='mse')
 
x = np.random.random((10000, 1))*100-50 #Get 10000 randon numbers between -50 + 50
y = x**2 # y = x^2
 
result = model.fit(x, y, epochs=1500, batch_size=256)
 
print(model.predict([4]))