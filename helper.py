# Importing all of the necessary libraries
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math

# Most of this code was recycled from previous projects

# Definition 
def date_parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')

# Definiton of the function for getting technical indicators
def get_technical_indicators(dataset, target_col):
    # Creates 7 and 21 day moving average
    dataset['ma7'] = dataset[target_col].rolling(window=7).mean()
    dataset['ma21'] = dataset[target_col].rolling(window=21).mean()

    # Creates the MACD: Provides exponential weighted functions.
    dataset['26ema'] = dataset[target_col].ewm(span=26).mean()
    dataset['12ema'] = dataset[target_col].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Creates the Bollinger Bands
    dataset['20sd'] = dataset[target_col].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + dataset['20sd'] * 2
    dataset['lower_band'] = dataset['ma21'] - dataset['20sd'] * 2

    # Creates the expoential moving average (EMA)
    dataset['ema'] = dataset[target_col].ewm(com=0.5).mean()

    # Creates the momentum
    dataset['momentum'] = dataset[target_col] - 1
    dataset['log_momentum'] = np.log(dataset[target_col] - 1)
    return dataset