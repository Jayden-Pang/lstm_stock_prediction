# Stock Prediction using LSTM

**Objective**
To test the viability of the Long Short Term Memory Model in predicting stock prices.

**Setting**
- Use 60 days of past close prices to predict the next day close price.
- Use 4 LSTM layers and 1 Dropout layer.

**Steps**
1) Obtain the relevant stock data from Yahoo Finance.
2) Split the data into 2 parts (70% training data, 30% test data) and normalize them.
3) Further split the data into arrays of 60 days close prices (predictor) and next day close price (response).
4) Build a sequential model with 4 LSTM layers, 1 Dropout layer, and 1 Dense layer.
5) Train the model using the training data and find the RMSE.
6) Plot the predicted close prices to visualize results.

