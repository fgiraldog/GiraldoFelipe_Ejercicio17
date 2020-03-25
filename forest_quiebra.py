
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score

atributes_name = pd.read_csv('atributes.txt')

atributes_name = list(atributes_name['atribute_name'])

data1 = pd.read_csv('1year.arff', sep=",", skiprows=69)
data2 = pd.read_csv('2year.arff', sep=",", skiprows=69)
data3 = pd.read_csv('3year.arff', sep=",", skiprows=69)
data4 = pd.read_csv('4year.arff', sep=",", skiprows=69)
data5 = pd.read_csv('5year.arff', sep=",", skiprows=69)

data1 = data1.to_numpy()
data2 = data2.to_numpy()
data3 = data3.to_numpy()
data4 = data4.to_numpy()
data5 = data5.to_numpy()

data = np.vstack((data1,data2))
data = np.vstack((data,data3))
data = np.vstack((data,data4))
data = np.vstack((data,data5))

data = np.delete(data, np.where(data == '?')[0], axis = 0)

data = np.array(data, dtype=float)

atributes = data[:,:-1]
target = data[:,-1]


x_train, x_testval, y_train, y_testval = sklearn.model_selection.train_test_split(atributes, target, test_size=0.5)

x_test, x_val, y_test, y_val = sklearn.model_selection.train_test_split(x_testval, y_testval, test_size=0.6)

n_trees = np.arange(1,200,25)
f1_train = []
f1_test = []

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))

# plt.figure()
# plt.scatter(n_trees,f1_train, label = 'train')
# plt.scatter(n_trees,f1_test, label = 'test')
# plt.legend()
# plt.show()
best_M = n_trees[np.argmax(f1_test)]
clf_best = sklearn.ensemble.RandomForestClassifier(n_estimators=best_M, max_features='sqrt')
clf_best.fit(x_train,y_train)
importances = clf_best.feature_importances_
f1_score = sklearn.metrics.f1_score(y_val, clf.predict(x_val))

a = pd.Series(importances, index=atributes_name)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title('{:.0f} Trees, F1 Score = {:.3f}'.format(best_M,f1_score))
plt.tight_layout()
plt.savefig('features.pngS')