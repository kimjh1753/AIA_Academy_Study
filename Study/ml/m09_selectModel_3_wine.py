from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답율 : ', accuracy_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')
'''        
  warnings.warn(message, FutureWarning)
AdaBoostClassifier 의 정답율 :  0.8321678321678322
BaggingClassifier 의 정답율 :  0.9090909090909091 
BernoulliNB 의 정답율 :  0.3916083916083916       
CalibratedClassifierCV 의 정답율 :  0.8041958041958042
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답율 :  0.36363636363636365   
ClassifierChain 은 없는 놈!
ComplementNB 의 정답율 :  0.7832167832167832
DecisionTreeClassifier 의 정답율 :  0.8811188811188811
DummyClassifier 의 정답율 :  0.35664335664335667      
ExtraTreeClassifier 의 정답율 :  0.8251748251748252
ExtraTreesClassifier 의 정답율 :  0.9090909090909091
GaussianNB 의 정답율 :  0.9090909090909091
GaussianProcessClassifier 의 정답율 :  0.3006993006993007
GradientBoostingClassifier 의 정답율 :  0.8321678321678322
HistGradientBoostingClassifier 의 정답율 :  0.3916083916083916
KNeighborsClassifier 의 정답율 :  0.7202797202797203
LabelPropagation 의 정답율 :  0.40559440559440557
LabelSpreading 의 정답율 :  0.40559440559440557
LinearDiscriminantAnalysis 의 정답율 :  0.965034965034965
LinearSVC 의 정답율 :  0.7902097902097902
LogisticRegression 의 정답율 :  0.8741258741258742
LogisticRegressionCV 의 정답율 :  0.8881118881118881
MLPClassifier 의 정답율 :  0.09090909090909091
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답율 :  0.916083916083916
NearestCentroid 의 정답율 :  0.6993006993006993
NuSVC 의 정답율 :  0.7902097902097902
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답율 :  0.6363636363636364
Perceptron 의 정답율 :  0.24475524475524477
QuadraticDiscriminantAnalysis 의 정답율 :  0.3916083916083916
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답율 :  0.8741258741258742
RidgeClassifier 의 정답율 :  0.965034965034965
RidgeClassifierCV 의 정답율 :  0.9090909090909091
SGDClassifier 의 정답율 :  0.24475524475524477
SVC 의 정답율 :  0.6713286713286714
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''

#wine의 Classifier 중에서 LinearDiscriminantAnalysis 의 정답율 :  0.965034965034965과 RidgeClassifier 의 정답율 :  0.965034965034965이 가장 좋다.

import sklearn
print(sklearn.__version__) # 0.23.2

