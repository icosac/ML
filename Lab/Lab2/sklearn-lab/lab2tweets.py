import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
cwd = os.getcwd()

XP_train=[]
file=open("./ocr/train-data.csv", 'r')
for l in file:
    app=[]
    for d in l.split(','):
        app.append(int(d))
    XP_train.append(app)
X_train=np.array(XP_train)

XP_test=[]
file=open("./ocr/test-data.csv", 'r')
for l in file:
    app=[]
    for d in l.split(','):
        app.append(int(d))
    XP_test.append(app)
X_test=np.array(XP_test)

YP_train=[]
file=open("./ocr/train-targets.csv", 'r')
for l in file:
    YP_train.append(l)
Y_train=np.array(YP_train)

YP_test=[]
file=open("./ocr/test-targets.csv", 'r')
for l in file:
    YP_test.append(l)
Y_test=np.array(YP_test)

x = X_test[2].reshape((16, 8))

plt.gray()
plt.matshow(x)
plt.show()

from sklearn.svm import SVC

clf =SVC(C=10, kernel='rbf', gamma=0.1)

# Training
clf.fit(X_train, Y_train)

# Prediction
Y_pred = clf.predict(X_test)

print(Y_pred)

from sklearn import metrics

report = metrics.classification_report(Y_test, Y_pred)

# the support is the number of instances having the given label in y_test
print(report)
print(metrics.accuracy_score(Y_test, Y_pred))
