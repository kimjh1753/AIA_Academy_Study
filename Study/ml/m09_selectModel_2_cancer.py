from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답을 : ', accuracy_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')
'''        
   warnings.warn(message, FutureWarning)
AdaBoostClassifier 의 정답을 :  0.9254385964912281
BaggingClassifier 의 정답을 :  0.9057017543859649
BernoulliNB 의 정답을 :  0.6359649122807017
CalibratedClassifierCV 의 정답을 :  0.9254385964912281
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답을 :  0.36403508771929827
ClassifierChain 은 없는 놈!
ComplementNB 의 정답을 :  0.9100877192982456
DecisionTreeClassifier 의 정답을 :  0.8662280701754386
DummyClassifier 의 정답을 :  0.5614035087719298
ExtraTreeClassifier 의 정답을 :  0.9210526315789473
ExtraTreesClassifier 의 정답을 :  0.9495614035087719
GaussianNB 의 정답을 :  0.9364035087719298
GaussianProcessClassifier 의 정답을 :  0.8596491228070176
GradientBoostingClassifier 의 정답을 :  0.8881578947368421
HistGradientBoostingClassifier 의 정답을 :  0.9451754385964912
KNeighborsClassifier 의 정답을 :  0.9057017543859649
LabelPropagation 의 정답을 :  0.37280701754385964
LabelSpreading 의 정답을 :  0.37280701754385964
LinearDiscriminantAnalysis 의 정답을 :  0.9517543859649122
LinearSVC 의 정답을 :  0.868421052631579
LogisticRegression 의 정답을 :  0.9364035087719298
LogisticRegressionCV 의 정답을 :  0.9451754385964912
MLPClassifier 의 정답을 :  0.36403508771929827
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답을 :  0.9122807017543859
NearestCentroid 의 정답을 :  0.8991228070175439
NuSVC 의 정답을 :  0.918859649122807
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답을 :  0.9276315789473685
Perceptron 의 정답을 :  0.868421052631579
QuadraticDiscriminantAnalysis 의 정답을 :  0.8925438596491229
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답을 :  0.9473684210526315
RidgeClassifier 의 정답을 :  0.9583333333333334
RidgeClassifierCV 의 정답을 :  0.9539473684210527
SGDClassifier 의 정답을 :  0.8289473684210527
SVC 의 정답을 :  0.9210526315789473
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
import sklearn
print(sklearn.__version__) # 0.23.2

