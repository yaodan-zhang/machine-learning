from load_mnist import load_mnist
from sklearn.neural_network import MLPClassifier
X_train, X_test, Y_train, Y_test = load_mnist(path = './', flatten = True, binary_data = False)
#Build CNN with Number of hidden layers chosen from {1, 2, 3}, Number of hidden nodes chosen from {50, 200, 784}.
for i in [1,2,3]:
    for j in [50,200,784]:
        CNN = MLPClassifier(hidden_layer_sizes = (j,)*i).fit(X_train, Y_train)
        print("For a CNN with ", i, " hidden layer(s) and ", j, " hidden nodes, the score of the performance on the testing data set is ", CNN.score(X_test, Y_test), '.', sep='')
