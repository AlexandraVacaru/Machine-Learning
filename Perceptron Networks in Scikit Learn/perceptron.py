from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

from sklearn.neural_network import MLPClassifier  # importul clasei
from sklearn import preprocessing

import numpy as np


# ------ exercitiul 1
def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[ y == -1, 0 ], X[ y == -1, 1 ], X[ y == -1, 2 ], 'b');
    ax.scatter3D(X[ y == 1, 0 ], X[ y == 1, 1 ], X[ y == 1, 2 ], 'r');
    plt.show()


def plot3d_data_and_decision_function(X, y, W, b):
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[ 0 ] * xx - W[ 1 ] * yy - b) / W[ 2 ]
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.scatter3D(X[ y == -1, 0 ], X[ y == -1, 1 ], X[ y == -1, 2 ], 'b');
    ax.scatter3D(X[ y == 1, 0 ], X[ y == 1, 1 ], X[ y == 1, 2 ], 'r');
    plt.show()


# incarcarea datelor de antrenare
X = np.loadtxt('./data/3d-points/x_train.txt')
y = np.loadtxt('./data/3d-points/y_train.txt', 'int')
y.astype(int)

# plot3d_data(X, y)
# incarcarea datelor de testare
X_test = np.loadtxt('./data/3d-points/x_test.txt')
y_test = np.loadtxt('./data/3d-points/y_test.txt', 'int') 

sc = preprocessing.StandardScaler()
sc.fit(X)
X_sc = sc.transform(X)
X_test_sc= sc.transform(X_test)

perceptron_model = Perceptron(eta0=0.1, tol=1e-5)

perceptron_model.fit(X_sc,y)
print("Acuratete pe multimea de antrenare: ", perceptron_model.score(X_sc, y))
print("Acuratete pe multimea de testare: ", perceptron_model.score(X_test_sc, y_test))

plot3d_data_and_decision_function(X_sc, y, np.squeeze(perceptron_model.coef_), perceptron_model.intercept_)

plot3d_data_and_decision_function(X_test_sc, y_test, np.squeeze(perceptron_model.coef_), perceptron_model.intercept_)

print(perceptron_model.n_iter_)
print(perceptron_model.coef_)
print(perceptron_model.intercept_)

# exercitiul 2

train_feats = np.loadtxt('data/MNIST/train_images.txt')
train_lbls = np.loadtxt('data/MNIST/train_labels.txt', 'int')

test_feats = np.loadtxt('data/MNIST/test_images.txt')
test_lbls = np.loadtxt('data/MNIST/test_labels.txt', 'int')

scaler = preprocessing.StandardScaler()

scaler.fit(train_feats)
train_feats_sc = scaler.transform(train_feats)
test_feats_sc = scaler.transform(test_feats)


def train_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print('Accuracy on train set is: ', model.score(X_train, y_train))
    print('Accuracy on test set is: ', model.score(X_test, y_test))
    print('Number of iterations until convergence is: ', model.n_iter_)

    print('\n')


# a)
clf = MLPClassifier(hidden_layer_sizes=(1,), activation='tanh', solver='sgd',learning_rate_init=0.01, momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# b)
clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', learning_rate_init=0.01, momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# c)
clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', learning_rate_init=0.00001, momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# d)
clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', learning_rate_init=10, momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# e)
clf = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', learning_rate_init=0.01, momentum=0,
                    max_iter=20)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# f)
clf = MLPClassifier(hidden_layer_sizes=(10, 10), activation='tanh', solver='sgd', learning_rate_init=0.01, momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# g)
clf = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='sgd', learning_rate_init=0.01, momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# h)
clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# i)
clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate_init=0.01,
                    momentum=0.9, max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# j)
clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate_init=0.01,
                    momentum=0.9, max_iter=2000, alpha=0.005)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)