# COMP 4112 Introduction to Data Science
# Assignment 2, Regression
# Francesco Coniglione (st#1206780)

"""
Read in the CSV dataset. You can do this how you like; Python lists are totally acceptable but
you might have to convert to other formats for scikit-learn sometimes. If you want, you could
use a pandas dataframe or a numpy array.
"""

import os, sys

import matplotlib.pyplot as plt
import numpy as np

# Importing pandas
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Experimenting with statsmodels for the F-Statistic and p-value
# Source(s): https://stackoverflow.com/questions/20701484/why-do-i-get-only-one-parameter-from-a-statsmodels-ols-fit, https://www.statsmodels.org/stable/regression.html
import statsmodels.api as sm

# Function to load data from CSV files
def data_loader(file_path):
    data = pd.read_csv(file_path, sep=';')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Loading data
wineRed_X, wineRed_y = data_loader(os.path.join(os.path.dirname(sys.argv[0]) + "/winequality-red.csv"))
wineWhite_X, wineWhite_y = data_loader(os.path.join(os.path.dirname(sys.argv[0]) + "/winequality-white.csv"))

# Split the data into training/testing sets
wineRed_X_train, wineRed_X_test, wineRed_y_train, wineRed_y_test = train_test_split(wineRed_X, wineRed_y, test_size=0.3)
wineWhite_X_train, wineWhite_X_test, wineWhite_y_train, wineWhite_y_test = train_test_split(wineWhite_X, wineWhite_y, test_size=0.3)

"""
Fit the three regression models using the LinearRegression from sklearn.
"""

# Create linear regression object
regr_red = linear_model.LinearRegression()
regr_white = linear_model.LinearRegression()

# Train the model using the training sets
regr_red.fit(wineRed_X_train, wineRed_y_train)
regr_white.fit(wineWhite_X_train, wineWhite_y_train)

# Define features for the consumer-only model (residual sugar (RS), pH, sulphates, alcohol)
wineRed_consumer_X = wineRed_X[:, [3, 8, 9, 10]]
wineWhite_consumer_X = wineWhite_X[:, [3, 8, 9, 10]]

# Split the consumer data into training/testing sets
wineRed_consumer_X_train, wineRed_consumer_X_test, wineRed_consumer_y_train, wineRed_consumer_y_test = train_test_split(wineRed_consumer_X, wineRed_y, test_size=0.3)
wineWhite_consumer_X_train, wineWhite_consumer_X_test, wineWhite_consumer_y_train, wineWhite_consumer_y_test = train_test_split(wineWhite_consumer_X, wineWhite_y, test_size=0.3)

# Train the consumer-only model
regr_consumer_red = linear_model.LinearRegression()
regr_consumer_red.fit(wineRed_consumer_X_train, wineRed_consumer_y_train)

regr_consumer_white = linear_model.LinearRegression()
regr_consumer_white.fit(wineWhite_consumer_X_train, wineWhite_consumer_y_train)

# Make predictions using the testing set
wineRed_y_pred = regr_red.predict(wineRed_X_test)
wineWhite_y_pred = regr_white.predict(wineWhite_X_test)
wineRed_consumer_y_pred = regr_consumer_red.predict(wineRed_consumer_X_test)
wineWhite_consumer_y_pred = regr_consumer_white.predict(wineWhite_consumer_X_test)

"""
Report on the performance of these models in Python with R^2 or MSE. Other measures can
be used as well.
"""

# Red Wine Model Performance
print("Red Wine Model Performance:")
print("Coefficients:", regr_red.coef_)
print("MSE: %.2f" % mean_squared_error(wineRed_y_test, wineRed_y_pred))
print("R^2: %.2f" % r2_score(wineRed_y_test, wineRed_y_pred))

wineRed_X_train_const = sm.add_constant(wineRed_X_train)
model_red = sm.OLS(wineRed_y_train, wineRed_X_train_const).fit()

print("Red Wine Model F-Statistic: %.2f" % model_red.fvalue)
print("Red Wine Model F-Statistic p-value: ", model_red.f_pvalue)

# Divider
print("-------------------")

# White Wine Model Performance
print("White Wine Model Performance:")
print("Coefficients:", regr_white.coef_)
print("MSE: %.2f" % mean_squared_error(wineWhite_y_test, wineWhite_y_pred))
print("R^2: %.2f" % r2_score(wineWhite_y_test, wineWhite_y_pred))

wineWhite_X_train_const = sm.add_constant(wineWhite_X_train)
model_white = sm.OLS(wineWhite_y_train, wineWhite_X_train_const).fit()

print("White Wine Model F-Statistic: %.2f" % model_white.fvalue)
print("White Wine Model F-Statistic p-value: ", model_white.f_pvalue)

# Divider
print("-------------------")

# Red Wine Consumer-Only Model Performance
print("Red Wine Consumer-Only Model Performance:")
print("Coefficients:", regr_consumer_red.coef_)
print("MSE: %.2f" % mean_squared_error(wineRed_consumer_y_test, wineRed_consumer_y_pred))
print("R^2: %.2f" % r2_score(wineRed_consumer_y_test, wineRed_consumer_y_pred))

wineRed_consumer_X_train_const = sm.add_constant(wineRed_consumer_X_train)
model_red_consumer = sm.OLS(wineRed_consumer_y_train, wineRed_consumer_X_train_const).fit()

print("Red Wine Consumer-Only Model F-Statistic: %.2f" % model_red_consumer.fvalue)
print("Red Wine Consumer-Only Model F-Statistic p-value: ", model_red_consumer.f_pvalue)

# Divider
print("-------------------")

# White Wine Consumer-Only Model Performance
print("White Wine Consumer-Only Model Performance:")
print("Coefficients:", regr_consumer_white.coef_)
print("MSE: %.2f" % mean_squared_error(wineWhite_consumer_y_test, wineWhite_consumer_y_pred))
print("R^2: %.2f" % r2_score(wineWhite_consumer_y_test, wineWhite_consumer_y_pred))

wineWhite_consumer_X_train_const = sm.add_constant(wineWhite_consumer_X_train)
model_white_consumer = sm.OLS(wineWhite_consumer_y_train, wineWhite_consumer_X_train_const).fit()

print("White Wine Consumer-Only Model F-Statistic: %.2f" % model_white_consumer.fvalue)
print("White Wine Consumer-Only Model F-Statistic p-value: ", model_white_consumer.f_pvalue)