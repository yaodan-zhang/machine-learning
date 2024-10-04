import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_excel("Classification iris.xlsx", header = None)
df.columns =['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']

df["Iris-setosa"] = pd.get_dummies(df['class'])["Iris-setosa"]
df["Iris-versicolor"] = pd.get_dummies(df['class'])["Iris-versicolor"]
df["Iris-virginica"] = pd.get_dummies(df['class'])["Iris-virginica"]

df.drop(["class"], axis=1, inplace = True)

for i in range(1,11):
    #shuffle the data
    df = shuffle(df).reset_index(drop=True)
    train = df.iloc[:120,:]
    test = df.iloc[120:,:]
    
    #build the model using formula calculated in 1.2, here k=3(setosa,versicolor,virginica)
    y1 = train["Iris-setosa"].to_numpy()
    y2 = train["Iris-versicolor"].to_numpy()
    y3 = train["Iris-virginica"].to_numpy()
    train.drop(["Iris-setosa","Iris-versicolor","Iris-virginica"], axis=1, inplace = True)
    X = train.to_numpy()
    X_tilt = np.concatenate((np.array([[1]]*len(train)), X), axis=1)
    w1 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_tilt), X_tilt)), np.matmul(np.transpose(X_tilt), y1))
    w2 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_tilt), X_tilt)), np.matmul(np.transpose(X_tilt), y2))
    w3 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_tilt), X_tilt)), np.matmul(np.transpose(X_tilt), y3))
    
    #calculate the classification error rate for training data
    train_predict = np.array([np.matmul(w1, np.transpose(X_tilt)),np.matmul(w2, np.transpose(X_tilt)),np.matmul(w3, np.transpose(X_tilt))])
    train_class_predict = np.argmax(train_predict, axis = 0) #predicted class
    train_class_truth = y2 + 2*y3 #underlying truth
    clf_err_train = np.count_nonzero(train_class_predict - train_class_truth)/len(train_class_truth)
    
    #calculate the classification error rate for testing data
    y2_test = test["Iris-versicolor"].to_numpy()
    y3_test = test["Iris-virginica"].to_numpy()
    test.drop(["Iris-setosa","Iris-versicolor","Iris-virginica"], axis=1, inplace = True)
    X_test = test.to_numpy()
    X_test_tilt = np.concatenate((np.array([[1]]*len(test)), X_test), axis=1)
    test_predict = np.array([np.matmul(w1, np.transpose(X_test_tilt)),np.matmul(w2, np.transpose(X_test_tilt)),np.matmul(w3, np.transpose(X_test_tilt))])
    test_class_predict = np.argmax(test_predict, axis = 0)
    test_class_truth = y2_test + 2*y3_test
    clf_err_test = np.count_nonzero(test_class_predict - test_class_truth)/len(test_class_truth)
    
    #report the classification error rates for both training and testing data sets
    print("For trial ", i, ", the classification error rate for training data set is ",\
        '%.4f'%clf_err_train, ", and classification error rate for testing data set is ", '%.4f'%clf_err_test, ".", sep='')
