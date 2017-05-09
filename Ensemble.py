
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas


train = pandas.read_csv("lab4-train.csv")
test = pandas.read_csv("lab4-test.csv")

array_train = train.values
array_test = test.values
X_train = array_train[:,0:4]
Y_train = array_train[:,4]
X_test = array_test[:,0:4]
Y_test = array_test[:,4]

seed = 200

#Task 1

#RandomForest Classifier
print("Random Forest Classifier:")
print('\n')

num_trees = 100
randomForest = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
randomForest = randomForest.fit(X_train,Y_train)
Y_predict = randomForest.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)
print("Accuracy with num_trees is 100: %0.4f" % (randomForest.score(X_test, Y_test)))
print("Confusion Matrix:")
print(Confusion_Mat)
print('\n')

num_trees = 130
randomForest = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
randomForest = randomForest.fit(X_train,Y_train)
Y_predict = randomForest.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)
print("Accuracy with num_trees is 130: %0.4f" % (randomForest.score(X_test, Y_test)))
print("Confusion Matrix:")
print(Confusion_Mat)
print('\n')

num_trees = 150
randomForest = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
randomForest = randomForest.fit(X_train,Y_train)
Y_predict = randomForest.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)
print("Accuracy with num_trees is 150: %0.4f" % (randomForest.score(X_test, Y_test)))
print("Confusion Matrix:")
print(Confusion_Mat)
print('\n')

#AdaBoost Classifier
print("AdaBoost Classifier:")
print('\n')

n_estimators = 130
learning_rate = 1.

dt_stump1 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump1 = dt_stump1.fit(X_train, Y_train)
dt_stump1_accuracy = dt_stump1.score(X_test, Y_test)

dt_stump2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
dt_stump2.fit(X_train, Y_train)
dt_stump2_accuracy = dt_stump2.score(X_test, Y_test)

adaBoost = AdaBoostClassifier(
                                  base_estimator=dt_stump1,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  random_state=seed,
                                  algorithm="SAMME")
adaBoost = adaBoost.fit(X_train, Y_train)
Y_predict = adaBoost.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)

print("Accuracy With dt_stump1 and n_estimators 130: %0.4f" % (adaBoost.score(X_test, Y_test)))
print("Confusion Matrix With dt_stump1:")
print(Confusion_Mat)
print('\n')

n_estimators = 150
adaBoost = AdaBoostClassifier(
                              base_estimator=dt_stump1,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              random_state=seed,
                              algorithm="SAMME")
adaBoost = adaBoost.fit(X_train, Y_train)
Y_predict = adaBoost.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)

print("Accuracy With dt_stump1 and n_estimators 150: %0.4f" % (adaBoost.score(X_test, Y_test)))
print("Confusion Matrix With dt_stump1:")
print(Confusion_Mat)
print('\n')

n_estimators = 130
adaBoost = AdaBoostClassifier(
                              base_estimator=dt_stump2,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              random_state=seed,
                              algorithm="SAMME")
adaBoost = adaBoost.fit(X_train, Y_train)
Y_predict = adaBoost.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)

print("Accuracy With dt_stump2 and n_estimators 130: %0.4f" % (adaBoost.score(X_test, Y_test)))
print("Confusion Matrix With dt_stump2:")
print(Confusion_Mat)
print('\n')

n_estimators = 150
adaBoost = AdaBoostClassifier(
                              base_estimator=dt_stump2,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              random_state=seed,
                              algorithm="SAMME")
adaBoost = adaBoost.fit(X_train, Y_train)
Y_predict = adaBoost.predict(X_test)
Confusion_Mat = confusion_matrix(Y_test, Y_predict)

print("Accuracy With dt_stump2 and n_estimators 150: %0.4f" % (adaBoost.score(X_test, Y_test)))
print("Confusion Matrix With dt_stump2:")
print(Confusion_Mat)
print('\n')

#Task 2
#Voting CLassifier
print("Ensemble Classifier:")
print('\n')

clf1 = LogisticRegression(random_state= seed)
clf1 = clf1.fit(X_train, Y_train)

clf2 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
clf2 = clf2.fit(X_train, Y_train)

clf3 = GaussianNB()
clf3 = clf3.fit(X_train, Y_train)

clf4 = KNeighborsClassifier(n_neighbors=10)
clf4 = clf4.fit(X_train, Y_train)

clf5 = MLPClassifier(random_state= seed)
clf5 = clf5.fit(X_train, Y_train)

for clf, label in zip([clf1, clf2, clf3, clf4, clf5], ['Logistic Regression', 'Decision Tree', 'naive Bayes', 'K Neighbours', 'Neural Network']):
    Y_predict = clf.predict(X_test)
    print("Accuracy: %0.4f [%s]" % (clf.score(X_test, Y_test), label))
    Confusion_Mat = confusion_matrix(Y_test, Y_predict)
    print("Confusion Matrix:")
    print(Confusion_Mat)

print('\n')

#Without weight
eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5)], voting='hard')
eclf = eclf.fit(X_train, Y_train)

eclf = eclf.fit(X_train, Y_train)
Y_predict = eclf.predict(X_test)
print("Accuracy Without Weights: %0.4f" % (eclf.score(X_test, Y_test)))

print('\n')

#Weighted Voting

eclf_weighted1 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5)], voting='soft',weights=[1,1,1,1,1])
eclf_weighted1 = eclf_weighted1.fit(X_train, Y_train)

Y_predict = eclf_weighted1.predict(X_test)
print("Accuracy with equal weights: %0.4f" % (eclf_weighted1.score(X_test, Y_test)))


eclf_weighted2 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5)], voting='soft',weights=[5,3,4,3,2])
eclf_weighted2 = eclf_weighted2.fit(X_train, Y_train)

Y_predict = eclf_weighted2.predict(X_test)
print("Accuracy with weights proportional to accuracy : %0.4f" % (eclf_weighted2.score(X_test, Y_test)))


eclf_weighted3 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5)], voting='soft',weights=[10,8,9,8,6])
eclf_weighted3 = eclf_weighted3.fit(X_train, Y_train)

Y_predict = eclf_weighted3.predict(X_test)
print("Accuracy with weights proportional to accuracy: %0.4f" % (eclf_weighted3.score(X_test, Y_test)))

print('\n')

#Task 3
#Include Random Forest and AdaBoost
print("Ensemble Classifier with RandomForest and AdaBoost:")
print('\n')

clf1 = LogisticRegression(random_state= seed)
clf1 = clf1.fit(X_train, Y_train)

clf2 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
clf2 = clf2.fit(X_train, Y_train)

clf3 = GaussianNB()
clf3 = clf3.fit(X_train, Y_train)

clf4 = KNeighborsClassifier(n_neighbors=10)
clf4 = clf4.fit(X_train, Y_train)

clf5 = MLPClassifier(random_state= seed)
clf5 = clf5.fit(X_train, Y_train)


num_trees = 130
randomForest = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
randomForest = randomForest.fit(X_train,Y_train)

n_estimators = 150
learning_rate = 1.
dt_stump1 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump1 = dt_stump1.fit(X_train, Y_train)
dt_stump1_accuracy = dt_stump1.score(X_test, Y_test)
adaBoost = AdaBoostClassifier(
                              base_estimator=dt_stump1,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              random_state=seed,
                              algorithm="SAMME")
adaBoost = adaBoost.fit(X_train, Y_train)

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, randomForest, adaBoost], ['Logistic Regression', 'Decision Tree', 'naive Bayes', 'K Neighbours', 'Neural Network', 'Random Forest', 'AdaBoost']):
    Y_predict = clf.predict(X_test)
    print("Accuracy: %0.4f [%s]" % (clf.score(X_test, Y_test), label))
    Confusion_Mat = confusion_matrix(Y_test, Y_predict)
    print("Confusion Matrix:")
    print(Confusion_Mat)

#Without weight

eclf2 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5), ('model', randomForest), ('ada', adaBoost)], voting='hard')
eclf2 = eclf2.fit(X_train, Y_train)

Y_predict = eclf2.predict(X_test)
print("Accuracy Without Weights: %0.4f" % (eclf2.score(X_test, Y_test)))

print('\n')

#Weighted Voting

eclf_weighted4 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[1,1,1,1,1,1,1])
eclf_weighted4 = eclf_weighted4.fit(X_train, Y_train)

Y_predict = eclf_weighted4.predict(X_test)
print("Accuracy with equal weights: %0.4f" % (eclf_weighted4.score(X_test, Y_test)))


eclf_weighted5 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[6,4,5,4,3,2,7])
eclf_weighted5 = eclf_weighted5.fit(X_train, Y_train)

Y_predict = eclf_weighted5.predict(X_test)
print("Accuracy with weights proportional to accuracy: %0.4f" % (eclf_weighted5.score(X_test, Y_test)))


eclf_weighted6 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[8,7,7,7,6,2,10])
eclf_weighted6 = eclf_weighted6.fit(X_train, Y_train)

Y_predict = eclf_weighted6.predict(X_test)
print("Accuracy with weights proportional to accuracy: %0.4f" % (eclf_weighted6.score(X_test, Y_test)))

eclf_weighted7 = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3), ('knn', clf4), ('nn', clf5), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[16,14,15,14,12,10,20])
eclf_weighted7 = eclf_weighted7.fit(X_train, Y_train)

Y_predict = eclf_weighted7.predict(X_test)
print("Accuracy with weights proportional to accuracy: %0.4f" % (eclf_weighted7.score(X_test, Y_test)))

print('\n')

