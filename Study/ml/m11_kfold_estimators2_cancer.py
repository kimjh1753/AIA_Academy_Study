from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # shuffled은 행을 섞는다

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답율 : \n', scores)
    except:
        # continue
        print(name, '은 없는 놈!')
'''        
  warnings.warn(message, FutureWarning)
AdaBoostClassifier 의 정답율 : 
 [0.91304348 0.95652174 0.91304348 0.86363636 0.86363636]
BaggingClassifier 의 정답율 : 
 [0.86956522 1.         0.82608696 0.90909091 1.        ]
BernoulliNB 의 정답율 :
 [0.43478261 0.30434783 0.26086957 0.59090909 0.5       ]
CalibratedClassifierCV 의 정답율 : 
 [0.95652174 0.86956522 0.86956522 0.72727273 0.86363636]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답율 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답율 : 
 [0.7826087  0.86956522 0.86956522 0.95454545 0.77272727]
DecisionTreeClassifier 의 정답율 :
 [1.         0.82608696 0.86956522 0.90909091 0.95454545]
DummyClassifier 의 정답율 :
 [0.60869565 0.52173913 0.56521739 0.54545455 0.68181818]
ExtraTreeClassifier 의 정답율 :
 [0.86956522 0.91304348 0.86956522 0.90909091 0.90909091]
ExtraTreesClassifier 의 정답율 : 
 [1.         1.         0.86956522 0.95454545 0.95454545]
GaussianNB 의 정답율 :
 [0.95652174 0.95652174 0.7826087  0.90909091 0.90909091]
GaussianProcessClassifier 의 정답율 : 
 [0.7826087  0.82608696 0.95652174 0.77272727 0.86363636]
GradientBoostingClassifier 의 정답율 : 
 [0.95652174 0.95652174 0.86956522 0.86363636 0.90909091]
HistGradientBoostingClassifier 의 정답율 : 
 [0.95652174 1.         0.86956522 0.86363636 1.        ]
KNeighborsClassifier 의 정답율 :
 [0.7826087  0.91304348 0.86956522 0.95454545 0.86363636]
LabelPropagation 의 정답율 :
 [0.47826087 0.17391304 0.52173913 0.40909091 0.45454545]
LabelSpreading 의 정답율 :
 [0.43478261 0.30434783 0.47826087 0.36363636 0.45454545]
LinearDiscriminantAnalysis 의 정답율 : 
 [0.95652174 0.95652174 0.82608696 0.86363636 0.86363636]
LinearSVC 의 정답율 : 
 [0.73913043 0.7826087  0.7826087  0.95454545 0.72727273]
LogisticRegression 의 정답율 : 
 [0.86956522 0.95652174 0.91304348 0.95454545 0.95454545]
LogisticRegressionCV 의 정답율 : 
 [0.91304348 0.95652174 0.95652174 0.81818182 0.95454545]
MLPClassifier 의 정답율 : 
 [0.82608696 0.95652174 0.82608696 0.31818182 0.72727273]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답율 :
 [0.86956522 0.86956522 0.82608696 0.81818182 0.86363636]
NearestCentroid 의 정답율 :
 [0.86956522 0.69565217 0.86956522 0.90909091 0.90909091]
NuSVC 의 정답율 : 
 [0.7826087  0.82608696 0.86956522 0.90909091 0.95454545]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답율 :
 [0.56521739 0.7826087  0.73913043 0.86363636 0.72727273]
Perceptron 의 정답율 :
 [0.52173913 0.86956522 0.13043478 0.86363636 0.5       ]
QuadraticDiscriminantAnalysis 의 정답율 :
 [0.91304348 0.91304348 0.91304348 0.90909091 1.        ]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답율 : 
 [0.82608696 0.95652174 0.95652174 1.         0.90909091]
RidgeClassifier 의 정답율 : 
 [0.91304348 0.95652174 0.95652174 1.         1.        ]
RidgeClassifierCV 의 정답율 :
 [0.95652174 0.95652174 0.91304348 1.         1.        ]
SGDClassifier 의 정답율 : 
 [0.82608696 0.65217391 0.82608696 0.81818182 0.81818182]
SVC 의 정답율 :
 [0.95652174 0.7826087  0.95652174 0.90909091 0.68181818]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
0.23.2
'''

import sklearn
print(sklearn.__version__) # 0.23.2

