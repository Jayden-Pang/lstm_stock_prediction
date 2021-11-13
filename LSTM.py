import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

def transform(data, step=1):
    x, y = [], []
    for i in range(step, len(data)):
        x.append(data[i - step:i, :])
        y.append(data[i, :])
    return np.array(x), np.array(y)

def rmse(real, predicted):
    return math.sqrt(mean_squared_error(real,predicted))

# Define Stock and Steps
ticker = "SNDL"
start_date = "2012-01-01"
step = 60

# Obtain Close
stock_full = yf.download(ticker,start = start_date)
stock = stock_full.iloc[:,4:5].values

# Normalize Data
training_size = int(len(stock)*0.7)
train = stock[:training_size,:]
test = stock[training_size:,:]
scaler_train = MinMaxScaler(feature_range=(0,1))
scaler_test = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler_train.fit_transform(train.reshape(-1,1))
scaled_test = scaler_test.fit_transform(test.reshape(-1,1))

# Separate Training Data and Test Data
x_train, y_train = transform (scaled_train, step)
x_test, y_test = transform (scaled_test, step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Create LSTM
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)

# Prediction
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Reverse Scale
train_predict = scaler_train.inverse_transform(train_predict)
test_predict = scaler_test.inverse_transform(test_predict)

# RMSE
train_RMSE = rmse(y_train,train_predict)
test_RMSE = rmse(y_test,test_predict)
print("Train RMSE = " + str(train_RMSE))
print("Test RMSE = " + str(test_RMSE))

# Visualize
plot_train = np.empty_like(stock)
plot_train[:,:] = np.nan
plot_train[step:len(train_predict)+step,:] = train_predict
plot_test = np.empty_like(stock)
plot_test[:,:] = np.nan
plot_test[(len(stock)-len(test_predict)):,:] = test_predict
plt.figure(figsize=(10,6))
plt.plot(stock, color = "black", label = "Actual Stock Price")
plt.plot(plot_test, color = "red", label = "Predicted Stock Price (Test Data)")
plt.plot(plot_train, color = "blue", label = "Predicted Stock Price (Training Data)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Visualize with Test Data Only
plt.figure(figsize=(10,6))
plot_full_test = stock[(len(stock)-len(test_predict)):,:]
plot_test_trimmed = plot_test[(len(stock)-len(test_predict)):,:]
plt.plot(plot_full_test, color = "Black",label = "Actual Stock Price")
plt.plot(plot_test_trimmed, color = "red", label = "Predicted Stock Price (Test Data)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()




