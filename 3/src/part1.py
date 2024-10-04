import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("Carseats.csv")

#Visualize Data Statistics Using Histograms
 #Numerical Data
hist = df.hist(bins=10)
plt.show()
 #Categorical Data
df['ShelveLoc'].value_counts()[['Good', 'Medium', 'Bad']].plot(kind='bar', title = "ShelveLoc")
plt.show()
df['Urban'].value_counts().plot(kind='bar', title = "Urban")
plt.show()
df['US'].value_counts().plot(kind='bar', title = "US")
plt.show()
plt.close()

#Convert Categorical Data into Numerical Ones and Process Training and Testing Sets
df["ShelveLoc"] = np.where(df["ShelveLoc"] == "Good", 2, 0) + np.where(df["ShelveLoc"] == "Medium", 1, 0) + np.where(df["ShelveLoc"] == "Bad", 0, 0)
df["Urban"] = np.where(df["Urban"] == "Yes", 1, 0)
df["US"] = np.where(df["US"] == "Yes", 1, 0)
train = df[:300]
test = df[300:]
X_train = np.array(train[["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]])
y_train = train["Sales"].values.flatten()
X_test = np.array(test[["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]])
y_test = test["Sales"].values.flatten()

#Decision Tree Regressor
#Repeat building the Tree for least node sizes = 3,5,10, maximum depths = 6,8,10.
for i in [3,5,10]:
    for j in [6,8,10]:
        DTR = DecisionTreeRegressor(random_state=0, max_depth=j, min_samples_leaf=i)
        DTR.fit(X_train, y_train)
        y_train_pred = DTR.predict(X_train)
        y_test_pred = DTR.predict(X_test)
        MSE_train = np.mean((y_train-y_train_pred)**2)
        MSE_test = np.mean((y_test-y_test_pred)**2)
        print("For Max Depth is ", j, ", Minimum Sample Leaf is ", i, ", the MSE for training set is ", \
        "%.4f" % MSE_train, " and MSE for testing set is ", "%.4f" % MSE_test, ".", sep = '')
        fig = plt.figure(figsize=(75,75))
        _ = tree.plot_tree(DTR, feature_names = df.columns.values[1:], filled = True)
        fig.savefig("DecistionTree_with_MaxDepth"+str(j)+"&MinSampleLeaf"+str(i)+".png")
        plt.close()
        
#Bagging for Decision Tree
for m in [7,9,11]:
    for n in range(10,110,10):
        BDTR = BaggingRegressor(base_estimator = DecisionTreeRegressor(random_state=0, max_depth=m), n_estimators=n, random_state=0).fit(X_train, y_train)
        y_train_pred = BDTR.predict(X_train)
        y_test_pred = BDTR.predict(X_test)
        MSE_train = np.mean((y_train-y_train_pred)**2)
        MSE_test = np.mean((y_test-y_test_pred)**2)
        print("For Max Depth is ", m, ", Number of Trees is ", n, ", the MSE for training set is ", \
        "%.4f" % MSE_train, " and MSE for testing set is ", "%.4f" % MSE_test, ".", sep = '')

#Random Forests Regressor
for n in range(10,110,10):
    for m in [3,4,5]:
        RFR = RandomForestRegressor(max_features= m, n_estimators = n, min_samples_split = 3, bootstrap = False, random_state = 0)
        RFR.fit(X_train, y_train)
        y_train_pred = RFR.predict(X_train)
        y_test_pred = RFR.predict(X_test)
        MSE_train = np.mean((y_train-y_train_pred)**2)
        MSE_test = np.mean((y_test-y_test_pred)**2)
        print("For Number of Features Considered is ", m, ", Number of Trees is ", n, ", the MSE for training set is ", \
        "%.4f" % MSE_train, " and MSE for testing set is ", "%.4f" % MSE_test, ".", sep = '')

#Plot Bias^2 w.r.t. Number of Trees, and Variance w.r.t. Number of Trees
#From previous part we observed m = 3 is the best, so we use 3 in this part.
Bias_2 = []
Variance = []
for n in range(10,410,10): # Iterate through the number of trees
    y_test_pred = np.array([0]*100)
    predicts = []
    for m in range(10):
        RFR2 = RandomForestRegressor(max_features= 3, n_estimators = n, min_samples_split = 3, bootstrap = False, random_state = 0)
        df = shuffle(df).reset_index(drop=True)
        train = df[:300]
        test = df[300:]
        X_train = np.array(train[["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]])
        y_train = train["Sales"].values.flatten()
        X_test = np.array(test[["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]])
        y_test = test["Sales"].values.flatten()
        RFR2.fit(X_train, y_train)
        pred = RFR2.predict(X_test)
        y_test_pred = y_test_pred + pred
        predicts.append(pred)
    Bias_2.append(np.mean((y_test_pred/10 - y_test)**2))
    Variance.append(np.mean(np.mean((np.array(predicts)-np.mean(np.array(predicts), axis=0))**2, axis=0)))
 
plt.plot(range(10,410,10), Bias_2)
plt.xlabel('Number of Trees')
plt.ylabel('Bias^2')
plt.title('Bias^2 versus Number of Trees in Random Forest Regressor')
plt.show()

plt.plot(range(10,410,10), Variance)
plt.xlabel('Number of Trees')
plt.ylabel('Variance')
plt.title('Variance versus Number of Trees in Random Forest Regressor')
plt.show()
