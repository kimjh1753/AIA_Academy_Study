from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
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
AdaBoostClassifier 의 정답을 :  0.9666666666666667
BaggingClassifier 의 정답을 :  0.9583333333333334
BernoulliNB 의 정답을 :  0.30833333333333335
CalibratedClassifierCV 의 정답을 :  0.875
CategoricalNB 의 정답을 :  0.9166666666666666
CheckingClassifier 의 정답을 :  0.375
ClassifierChain 은 없는 놈!
ComplementNB 의 정답을 :  0.6833333333333333
DecisionTreeClassifier 의 정답을 :  0.9666666666666667
DummyClassifier 의 정답을 :  0.4
ExtraTreeClassifier 의 정답을 :  0.9
ExtraTreesClassifier 의 정답을 :  0.9583333333333334
GaussianNB 의 정답을 :  0.9416666666666667
GaussianProcessClassifier 의 정답을 :  0.925
GradientBoostingClassifier 의 정답을 :  0.95
HistGradientBoostingClassifier 의 정답을 :  0.30833333333333335
KNeighborsClassifier 의 정답을 :  0.9666666666666667
LabelPropagation 의 정답을 :  0.9416666666666667
LabelSpreading 의 정답을 :  0.9416666666666667
LinearDiscriminantAnalysis 의 정답을 :  0.9666666666666667
LinearSVC 의 정답을 :  0.9666666666666667
LogisticRegression 의 정답을 :  0.925
LogisticRegressionCV 의 정답을 :  0.9
MLPClassifier 의 정답을 :  1.0
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답을 :  0.8916666666666667
NearestCentroid 의 정답을 :  0.875
NuSVC 의 정답을 :  0.9166666666666666
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답을 :  0.85
Perceptron 의 정답을 :  0.55
QuadraticDiscriminantAnalysis 의 정답을 :  0.9583333333333334
RadiusNeighborsClassifier 의 정답을 :  0.9166666666666666
RandomForestClassifier 의 정답을 :  0.9583333333333334
RidgeClassifier 의 정답을 :  0.7916666666666666
RidgeClassifierCV 의 정답을 :  0.7583333333333333
SGDClassifier 의 정답을 :  0.6833333333333333
SVC 의 정답을 :  0.9
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
import sklearn
print(sklearn.__version__) # 0.23.2

