'''
Course:  CEG 4195
Name:    Henry Li

Running instructions:
1. Open a command prompt.
2. Enter "conda init"
3. Enter "conda activate <environment name>"
4. Enter "python assignment<#>.py" to run the code. 

API/dataset: "yfinance"
'''

#=============================================================================
#                              Imports
#=============================================================================
import yfinance as yf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#=============================================================================
#                              Data Collection
#=============================================================================
ticker = "AAPL"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")

#=============================================================================
#                              Data Preprocessing
#=============================================================================
# I want to work with daily data.
# Data is already daily, so no further resampling is required.

# yfinance only includes trading days. 
# Weekends and holiday dates don't appear in the dataset.
# No need to fill or remove null values for these days.

prices = data['Close'].squeeze()          
prices = prices.dropna()                    # Handles data that is "Not a number" (NaN), that is null values.

returns = prices.pct_change()               # Daily returns (percentage change) in the stock.
returns = returns.dropna()                  

#=============================================================================
#                              Model Fitting
#=============================================================================
# Idea: Predict the current day's return (y) based on yesterday's return (X).

# Features (X) - Yesterday's return.
X = returns.shift(1).dropna().values.reshape(-1,1) 

# Targets (y) - Today's return.
y = returns[1:].values                                  # Aligning today's return with yesterday's return.

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train,y_train)

# Evaluate model
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)

#=============================================================================
#                       Financial Metric Computation
#=============================================================================
# Assume 252 trading days.

# Annual return 
mean_daily_return = returns.mean()
annual_return = (1+mean_daily_return)**252-1
print("Annual Return: ", annual_return)

# Sharpe ratio
risk_free_rate = 0.02                       # Choose a rate of 2%
sharpe_ratio = (returns.mean()*252 - risk_free_rate)/(returns.std()*np.sqrt(252))
print("Sharpe Ratio: ", sharpe_ratio)

# Sortino ratio
downside_returns = returns[returns<0]
sortino_ratio = (returns.mean()*252 - risk_free_rate)/(downside_returns.std()*np.sqrt(252))
print("Sortino Ratio: ", sortino_ratio)