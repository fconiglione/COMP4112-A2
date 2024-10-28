#A2 skeleton program - you only needed to make minor modifications to the regression python examples

import os, sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#I copied and pasted the CSV load code from the regression examples. It would be better
#to read these in with a function, pandas dataframe, so a refactor of this code is good for a submission.

wineRed_X = []
wineRed_y = []
lnSkip = 2
with open(os.path.join(os.path.dirname(sys.argv[0]) + "/winequality-red.csv")) as file:
    for line in file:
        if lnSkip == 0:
            ln = line.rstrip().split(";")

            #need to do edit the lists below to read in the full CSV
            ln_data_X = [float(ln[0]), float(ln[1])]

            #use last column for class label
            ln_data_y = [float(ln[len(ln) - 1])]
            wineRed_X.append(ln_data_X)
            wineRed_y.append(ln_data_y)
        else:
            lnSkip = lnSkip - 1

wineWhite_X = []
wineWhite_y = []
lnSkip = 2
with open(os.path.join(os.path.dirname(sys.argv[0]) + "/winequality-white.csv")) as file:
    for line in file:
        if lnSkip == 0:
            ln = line.rstrip().split(";")

            #need to edit the lists below to read in the full CSV
            ln_data_X = [float(ln[0]), float(ln[1])]

            #use last column for classs label
            ln_data_y = [float(ln[len(ln) - 1])]
            wineWhite_X.append(ln_data_X)
            wineWhite_y.append(ln_data_y)
        else:
            lnSkip = lnSkip - 1

# Split the data into training/testing sets
wineRed_X_train, wineRed_X_test, wineRed_y_train, wineRed_y_test = train_test_split(wineRed_X, wineRed_y, test_size=0.3)

# Split the data into training/testing sets
wineWhite_X_train, wineWhite_X_test, wineWhite_y_train, wineWhite_y_test = train_test_split(wineWhite_X, wineWhite_y,test_size=0.3)


# can train models below


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(wineRed_X_train, wineRed_y_train)

# Make predictions using the testing set
wineRed_y_pred = regr.predict(wineRed_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("MSE: %.2f" % mean_squared_error(wineRed_y_test, wineRed_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("R^2: %.2f" % r2_score(wineRed_y_test, wineRed_y_pred))

