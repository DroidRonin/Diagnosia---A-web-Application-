# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:21:17 2017

@author: Jess
"""

from pandas import read_csv
from pandas import datetime
import pandas as pd
from pandas import DataFrame

from matplotlib import pyplot
from matplotlib.pylab import rcParams
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error



rcParams['figure.figsize'] = 15, 6


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m')
series = pd.read_csv('s1-table.csv', parse_dates=['date'] , index_col=['date'],date_parser=dateparse)
print(series.head())
series.plot()
pyplot.show()
 
autocorrelation_plot(series)
pyplot.show()
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

"""FORECAST""" 
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()



