# Arima intuition 

ARIMA, which stands for Autoregressive Integrated Moving Average, combines three models - Autoregressive (AR), Integrated (I), and Moving Average (MA). Each model contributes differently to the prediction of the current value in a time series, and the parameters for all three models are estimated using Maximum Likelihood Estimation (MLE).

1. Autoregressive (AR) model represents the relationship between the current observation and its previous observations.
2. Integrated (I) model represents the differencing of the original time series to make it stationary.
3. Moving Average (MA) model represents the relationship between the current observation and the residual errors obtained from previous observations.

By estimating the parameters using MLE, we can combine these models optimally and weigh their contributions to predict the current value in a time series effectively.
