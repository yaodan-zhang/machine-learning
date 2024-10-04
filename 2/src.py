from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

train = np.loadtxt("train.txt")
test = np.loadtxt("test.txt")

################################################################
#######               1.The linear model             ###########
################################################################
f1 = open("SVM_linear.txt", "x")

Clf1 = OneVsRestClassifier(SVC(C = 1e5, kernel="linear"))

X = train[:,[1,2,3,4]]
y = train[:,[0]].ravel()
Z = test[:,[1,2,3,4]]
k = test[:,[0]].ravel()

Clf1.fit(X, y)

train_err_1 = np.count_nonzero(Clf1.predict(X)-y)/y.size
test_err_1 = np.count_nonzero(Clf1.predict(Z)-k)/k.size

f1.write("training_error:"+str(train_err_1)+"\n")
f1.write("testing_error:"+str(test_err_1)+"\n")

f1.write("w_of_setosa:"+','.join(map(str,Clf1.estimators_[0].coef_.ravel()))+"\n")
f1.write("b_of_setosa:"+str(Clf1.estimators_[0].intercept_[0])+"\n")
f1.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf1.estimators_[0].support_))+"\n")

f1.write("w_of_versicolor:"+','.join(map(str,Clf1.estimators_[1].coef_.ravel()))+"\n")
f1.write("b_of_versicolor:"+str(Clf1.estimators_[1].intercept_[0])+"\n")
f1.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf1.estimators_[1].support_))+"\n")

f1.write("w_of_virginica:"+','.join(map(str,Clf1.estimators_[2].coef_.ravel()))+"\n")
f1.write("b_of_virginica:"+str(Clf1.estimators_[2].intercept_[0])+"\n")
f1.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf1.estimators_[2].support_))+"\n")

f1.close()

print("Class setosa linearly separable:",np.count_nonzero(Clf1.estimators_[0].predict(X)-np.where(y==0,1,0))==0)
print("Class versicolor linearly separable:",np.count_nonzero(Clf1.estimators_[1].predict(X)-np.where(y==1,1,0))==0)
print("Class virginica linearly separable:",np.count_nonzero(Clf1.estimators_[2].predict(X)-np.where(y==2,1,0))==0)

################################################################
#######                2.Slack variables               #########
################################################################
f2 = open("SVM_slack.txt", "x")

for i in range(1,11):
    C = 0.1 * i
    
    Clf2 = OneVsRestClassifier(SVC(C = C, kernel="linear"))
    Clf2.fit(X, y)
    
    train_err_2 = np.count_nonzero(Clf2.predict(X)-y)/y.size
    test_err_2 = np.count_nonzero(Clf2.predict(Z)-k)/k.size
    
    slack1 = (1 - np.where(y==0,1,-1) * (np.matmul(X,np.transpose(Clf2.estimators_[0].coef_))+Clf2.estimators_[0].intercept_[0]).ravel())
    slack1[slack1<0]=0
    slack2 = (1 - np.where(y==1,1,-1) * (np.matmul(X,np.transpose(Clf2.estimators_[1].coef_))+Clf2.estimators_[1].intercept_[0]).ravel())
    slack2[slack2<0]=0
    slack3 = (1 - np.where(y==2,1,-1) * (np.matmul(X,np.transpose(Clf2.estimators_[2].coef_))+Clf2.estimators_[2].intercept_[0]).ravel())
    slack3[slack3<0]=0
    
    f2.write("C="+str(C)+"\n")
    f2.write("training_error:"+str(train_err_2)+"\n")
    f2.write("testing_error:"+str(test_err_2)+"\n")
    
    f2.write("w_of_setosa:"+','.join(map(str,Clf2.estimators_[0].coef_.ravel()))+"\n")
    f2.write("b_of_setosa:"+str(Clf2.estimators_[0].intercept_[0])+"\n")
    f2.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf2.estimators_[0].support_))+"\n")
    f2.write("slack_variable_of_setosa:"+','.join(map(str,slack1))+"\n")
    
    f2.write("w_of_versicolor:"+','.join(map(str,Clf2.estimators_[1].coef_.ravel()))+"\n")
    f2.write("b_of_versicolor:"+str(Clf2.estimators_[1].intercept_[0])+"\n")
    f2.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf2.estimators_[1].support_))+"\n")
    f2.write("slack_variable_of_versicolor:"+','.join(map(str,slack2))+"\n")
    
    f2.write("w_of_virginica:"+','.join(map(str,Clf2.estimators_[2].coef_.ravel()))+"\n")
    f2.write("b_of_virginica:"+str(Clf2.estimators_[2].intercept_[0])+"\n")
    f2.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf2.estimators_[2].support_))+"\n")
    f2.write("slack_variable_of_virginica:"+','.join(map(str,slack3))+"\n")
    
    f2.write("\n")
f2.close()

################################################################
#######  3.kernel function poly2 with slack variables  #########
################################################################
f3 = open("SVM_poly2.txt", "x")

Clf3 = OneVsRestClassifier(SVC(C = 1, kernel="poly", degree = 2))
Clf3.fit(X, y)

train_err_3 = np.count_nonzero(Clf3.predict(X)-y)/y.size
test_err_3 = np.count_nonzero(Clf3.predict(Z)-k)/k.size

f3.write("training_error:"+str(train_err_3)+"\n")
f3.write("testing_error:"+str(test_err_3)+"\n")

f3.write("b_of_setosa:"+str(Clf3.estimators_[0].intercept_[0])+"\n")
f3.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf3.estimators_[0].support_))+"\n")

f3.write("b_of_versicolor:"+str(Clf3.estimators_[1].intercept_[0])+"\n")
f3.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf3.estimators_[1].support_))+"\n")

f3.write("b_of_virginica:"+str(Clf3.estimators_[2].intercept_[0])+"\n")
f3.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf3.estimators_[2].support_))+"\n")

f3.close()

################################################################
#######   4.kernel function poly3 with slack variables  ########
################################################################
f4 = open("SVM_poly3.txt", "x")

Clf4 = OneVsRestClassifier(SVC(C = 1, kernel="poly", degree = 3))
Clf4.fit(X, y)

train_err_4 = np.count_nonzero(Clf4.predict(X)-y)/y.size
test_err_4 = np.count_nonzero(Clf4.predict(Z)-k)/k.size

f4.write("training_error:"+str(train_err_4)+"\n")
f4.write("testing_error:"+str(test_err_4)+"\n")

f4.write("b_of_setosa:"+str(Clf4.estimators_[0].intercept_[0])+"\n")
f4.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf4.estimators_[0].support_))+"\n")

f4.write("b_of_versicolor:"+str(Clf4.estimators_[1].intercept_[0])+"\n")
f4.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf4.estimators_[1].support_))+"\n")

f4.write("b_of_virginica:"+str(Clf4.estimators_[2].intercept_[0])+"\n")
f4.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf4.estimators_[2].support_))+"\n")

f4.close()

#####################################################################
######   5.kernel function rbf (sigma=1) with slack variables  ######
#####################################################################
f5 = open("SVM_rbf.txt", "x")

Clf5 = OneVsRestClassifier(SVC(C = 1, kernel="rbf", gamma = 0.5))
Clf5.fit(X, y)

train_err_5 = np.count_nonzero(Clf5.predict(X)-y)/y.size
test_err_5 = np.count_nonzero(Clf5.predict(Z)-k)/k.size

f5.write("training_error:"+str(train_err_5)+"\n")
f5.write("testing_error:"+str(test_err_5)+"\n")

f5.write("b_of_setosa:"+str(Clf5.estimators_[0].intercept_[0])+"\n")
f5.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf5.estimators_[0].support_))+"\n")

f5.write("b_of_versicolor:"+str(Clf5.estimators_[1].intercept_[0])+"\n")
f5.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf5.estimators_[1].support_))+"\n")

f5.write("b_of_virginica:"+str(Clf5.estimators_[2].intercept_[0])+"\n")
f5.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf5.estimators_[2].support_))+"\n")

f5.close()

######################################################################
####   6.kernel function sigmoid (sigma=1) with slack variables   ####
######################################################################
f6 = open("SVM_sigmoid.txt", "x")

Clf6 = OneVsRestClassifier(SVC(C = 1, kernel="sigmoid", gamma = 0.5))
Clf6.fit(X, y)

train_err_6 = np.count_nonzero(Clf6.predict(X)-y)/y.size
test_err_6 = np.count_nonzero(Clf6.predict(Z)-k)/k.size

f6.write("training_error:"+str(train_err_6)+"\n")
f6.write("testing_error:"+str(test_err_6)+"\n")

f6.write("b_of_setosa:"+str(Clf6.estimators_[0].intercept_[0])+"\n")
f6.write("support_vector_indices_of_setosa:"+','.join(map(str,Clf6.estimators_[0].support_))+"\n")

f6.write("b_of_versicolor:"+str(Clf6.estimators_[1].intercept_[0])+"\n")
f6.write("support_vector_indices_of_versicolor:"+','.join(map(str,Clf6.estimators_[1].support_))+"\n")

f6.write("b_of_virginica:"+str(Clf6.estimators_[2].intercept_[0])+"\n")
f6.write("support_vector_indices_of_virginica:"+','.join(map(str,Clf6.estimators_[2].support_))+"\n")

f6.close()
