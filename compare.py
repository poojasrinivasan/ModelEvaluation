import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import precision_score,recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import logging
LOG_FILENAME = 'Evaluation.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)


data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",header=None)
# data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",header=None)
# data.columns = ["buying","maint","doors","persons","lug_boot","safety","Targetclass"]
# le = preprocessing.LabelEncoder()
# data = data.apply(le.fit_transform)
#le = preprocessing.LabelEncoder()
#data = data.apply(le.fit_transform)
#splitting data and target values
Count_Row=data.shape[0] #gives number of row count
Count_Col=data.shape[1]
X=data.iloc[:,0:Count_Col-1]
y=data.iloc[:,Count_Col-1]
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X= X.as_matrix(columns=None)
y = y.as_matrix(columns=None)
crossValidationFold = 10
kf = KFold(crossValidationFold)
kf.get_n_splits(X)


##***************************************************Neural Network********************************************
print("*********Neural Network**************")
#Neural Network
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    hLSize = [10,10]
    solver = 'lbfgs'
    lRate = 'adaptive'
    activation = 'tanh'
    itr = 2000
    mlp = MLPClassifier(hidden_layer_sizes=hLSize,solver=solver,learning_rate=lRate,max_iter=itr,random_state=5,activation=activation)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    accuracy += mlp.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
logging.info( '\n \t Classifier : Neural Network' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Hidden Layer Size: ' + str(hLSize) + ' , Activation: ' + activation +' , solver : ' + str(solver) +
              ' , Learning Rate : ' + str(lRate) + ' , Iteration : ' + str(itr) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Decision Tree********************************************
print("*********Decision Tree**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    criterion = 'entropy'
    maxdepth=5
    DT = DecisionTreeClassifier(criterion=criterion, random_state=1,
                                max_depth=maxdepth, min_samples_leaf=1)
    DT.fit(X_train, y_train)
    predictions = DT.predict(X_test)
    accuracy += DT.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Decision Tree' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Criterion: ' + criterion +
              ' , Max Depth: ' + str(maxdepth)+
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Perceptron********************************************
print("*********Perceptron**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    penalty = "l1"
    iteration = 1000
    alpha1=0.01
    PT = Perceptron(penalty, n_iter=iteration,alpha=alpha1)
    PT.fit(X_train, y_train)
    predictions = PT.predict(X_test)
    accuracy += PT.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Perceptron' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Penalty: ' + penalty + ' , Iteration : ' + str(iteration) +
             ' , alpha : ' + str(alpha1) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))


##***************************************************SVM********************************************
print("*********SVM**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ker='rbf'
    svm_clf = svm.SVC(kernel = ker)
    svm_clf.fit(X_train, y_train)
    predictions = svm_clf.predict(X_test)
    accuracy += svm_clf.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)


logging.info( '\n \t Classifier : SVM' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Kernel: ' + ker +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))


##***************************************************Naive Bayes********************************************
print("*********Naive Bayes**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    accuracy += nb.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Naive Bayes' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Logistic Regression********************************************
print("*********Logistic Regression**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    penalty = 'l2'
    solver = 'sag'
    lr = LogisticRegression(penalty= penalty, solver= solver)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    accuracy += lr.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Logistic Regression' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Penalty: ' + penalty + ' , Solver: ' + solver +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))


##***************************************************Nearest Neighbour********************************************
print("*********Nearest Neighbour**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    k =10
    nn = KNeighborsClassifier(n_neighbors=k)
    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)
    accuracy += nn.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Nearest Neighbour' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Value of K: ' + str(k) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Bagging********************************************
print("*********Bagging**************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimators = 20
    max_feature=2
    bag = BaggingClassifier(n_estimators=estimators,max_features=max_feature)
    bag.fit(X_train, y_train)
    predictions = bag.predict(X_test)
    accuracy += bag.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Bagging' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' ,No of estimators: ' + str(estimators) +
              ' ,max_features ' + str(max_feature) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))


##***************************************************Random Forest********************************************
print("*********Random Forest*************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimators = 15
    crit = 'entropy'
    rf = RandomForestClassifier(n_estimators=estimators,max_features=2, criterion=crit)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy += rf.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Random Forest' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Estimators : ' + str(estimators) + ' , Criterion : ' + str(crit) +
              ' ,max_features ' + str(max_feature) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************ADA Boost********************************************
print("*********Ada Boost*************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimators = 25
    lRate =0.5
    ab = AdaBoostClassifier(n_estimators=estimators, learning_rate=lRate)
    ab.fit(X_train, y_train)
    predictions = ab.predict(X_test)
    accuracy += ab.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Ada Boost' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Estimator: ' + str(estimators) + ' , Learning Rate: ' + str(lRate) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Gradient Boosting********************************************
print("*********Gradient Boosting*************")
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    loss= 'deviance'
    lRate = 0.5
    estimators = 10
    gb = GradientBoostingClassifier(loss=loss, learning_rate=lRate, n_estimators=estimators)
    gb.fit(X_train, y_train)
    predictions = gb.predict(X_test)
    accuracy += gb.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

logging.info( '\n \t Classifier : Gradient Boosting' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Loss : ' + loss + ' , Learning Rate : ' + str(lRate) + ' , Estimators : ' + str(estimators) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))

##***************************************************Neural Network********************************************
print("*********Deep Learning**************")
#Neural Network
accuracy = 0
precision = 0
recall = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    hLSize = [15,15,15]
    solver = 'adam'
    lRate = 'adaptive'
    activation = 'logistic'
    itr = 1000
    mlp = MLPClassifier(hidden_layer_sizes=hLSize,solver=solver,learning_rate=lRate,max_iter=itr,random_state=5,activation=activation)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    accuracy += mlp.score(X_test, y_test)
    precision += precision_score(y_test,predictions)
    recall += recall_score(y_test,predictions)
accuracy = accuracy/10
precision = precision/10
recall = recall/10

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
logging.info( '\n \t Classifier : DeepLearning' + ' , CrossValidationFold : ' + str(crossValidationFold) +
              ' , Hidden Layer Size: ' + str(hLSize) + ' , Activation: ' + activation +' , solver : ' + str(solver) +
              ' , Learning Rate : ' + str(lRate) + ' , Iteration : ' + str(itr) +
              ' , Average Accuracy : ' + str(accuracy) + ' , Average Precision :' + str(precision) + ' , Average Recall : ' + str(recall))
