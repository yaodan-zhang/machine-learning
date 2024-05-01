import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle

df1 = pd.read_csv('Regression.csv')
df1 = df1.dropna(axis = 0).reset_index(drop = True) #drop rows with NaN values
df1.drop(["station","Date"], axis=1, inplace = True) #drop the first 2 columns
for i in range(1,11):
    #shuffle the data
    df1 = shuffle(df1).reset_index(drop=True)
    train = df1.iloc[:6070,:]
    test = df1.iloc[6070:,:]
    #build the model using formula calculated in 1.2, here k=2(Next_Tmax,Next_Tmin)
    y1 = train["Next_Tmax"].to_numpy()
    y2 = train["Next_Tmin"].to_numpy()
    train.drop(["Next_Tmax","Next_Tmin"], axis=1, inplace = True)
    X = train.to_numpy()
    X_tilt = np.concatenate((np.array([[1]]*len(train)), X), axis=1)
    w1 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_tilt), X_tilt)), np.matmul(np.transpose(X_tilt), y1))
    w2 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_tilt), X_tilt)), np.matmul(np.transpose(X_tilt), y2))
    #calculate the RMSE for training error
    err1_train = np.square(y1 - np.matmul(w1, np.transpose(X_tilt))).sum()
    err2_train = np.square(y2 - np.matmul(w2, np.transpose(X_tilt))).sum()
    RMSE_train = math.sqrt((err1_train + err2_train) / len(train))
    #calculate the RMSE for testing error
    y1_test = test["Next_Tmax"].to_numpy()
    y2_test = test["Next_Tmin"].to_numpy()
    test.drop(["Next_Tmax","Next_Tmin"], axis=1, inplace = True)
    X_test = test.to_numpy()
    X_test_tilt = np.concatenate((np.array([[1]]*len(test)), X_test), axis=1)
    err1_test = np.square(y1_test - np.matmul(w1,np.transpose(X_test_tilt))).sum()
    err2_test = np.square(y2_test - np.matmul(w2,np.transpose(X_test_tilt))).sum()
    RMSE_test = math.sqrt((err1_test + err2_test) / len(test))
    #report the training and testing RMSEs
    print("For round ", i, ", the RMSE for training error is ", RMSE_train,\
    ", RMSE for testing error is ", RMSE_test, ".", sep='')
