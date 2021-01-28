from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
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
 [0.83333333 0.83333333 1.         1.         0.66666667]
BaggingClassifier 의 정답율 : 
 [1.         1.         0.83333333 0.83333333 0.83333333]
BernoulliNB 의 정답율 :
 [0.16666667 0.5        0.16666667 0.33333333 0.16666667]
CalibratedClassifierCV 의 정답율 : 
 [0.83333333 0.66666667 0.83333333 0.83333333 1.        ]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답율 : 
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답율 :
 [0.5        0.83333333 0.83333333 0.33333333 0.5       ]
DecisionTreeClassifier 의 정답율 : 
 [1.         0.83333333 0.83333333 1.         1.        ]
DummyClassifier 의 정답율 :
 [0.66666667 0.5        0.5        0.33333333 0.33333333]
ExtraTreeClassifier 의 정답율 :
 [0.83333333 0.83333333 1.         1.         1.        ]
ExtraTreesClassifier 의 정답율 : 
 [0.83333333 1.         1.         0.83333333 1.        ]
GaussianNB 의 정답율 :
 [1.         0.83333333 1.         1.         1.        ]
GaussianProcessClassifier 의 정답율 : 
 [1.         1.         1.         1.         0.83333333]
GradientBoostingClassifier 의 정답율 : 
 [1.         1.         0.83333333 0.83333333 0.83333333]
HistGradientBoostingClassifier 의 정답율 : 
 [0.33333333 0.33333333 0.33333333 0.16666667 0.33333333]
KNeighborsClassifier 의 정답율 :
 [1.         0.66666667 0.83333333 0.83333333 1.        ]
LabelPropagation 의 정답율 :
 [1.         0.83333333 0.83333333 0.66666667 1.        ]
LabelSpreading 의 정답율 :
 [0.66666667 0.83333333 1.         1.         0.83333333]
LinearDiscriminantAnalysis 의 정답율 : 
 [0.83333333 0.83333333 1.         0.83333333 1.        ]
LinearSVC 의 정답율 : 
 [0.83333333 1.         1.         0.66666667 0.83333333]
LogisticRegression 의 정답율 : 
 [0.83333333 0.83333333 0.83333333 1.         1.        ]
LogisticRegressionCV 의 정답율 : 
 [1.         0.83333333 0.83333333 1.         0.66666667]
MLPClassifier 의 정답율 : 
 [0.66666667 1.         0.83333333 0.83333333 1.        ]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답율 :
 [0.33333333 0.33333333 0.5        0.83333333 1.        ]
NearestCentroid 의 정답율 :
 [1.         1.         1.         0.83333333 1.        ]
NuSVC 의 정답율 : 
 [       nan 0.83333333 0.83333333 0.83333333        nan]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답율 : 
 [0.5        0.66666667 0.33333333 0.66666667 0.33333333]
Perceptron 의 정답율 : 
 [0.66666667 0.33333333 0.66666667 0.5        0.83333333]
QuadraticDiscriminantAnalysis 의 정답율 : 
 [0.83333333 1.         0.5        0.16666667 0.5       ]
RadiusNeighborsClassifier 의 정답율 :
 [0.83333333 0.83333333 1.         1.         0.83333333]
RandomForestClassifier 의 정답율 : 
 [0.83333333 1.         0.83333333 0.83333333 1.        ]
RidgeClassifier 의 정답율 :
 [1.         0.83333333 0.66666667 0.66666667 0.83333333]
RidgeClassifierCV 의 정답율 : 
 [0.83333333 0.66666667 0.66666667 0.66666667 1.        ]
SGDClassifier 의 정답율 : 
 [0.16666667 0.66666667 0.33333333 0.83333333 0.33333333]
SVC 의 정답율 :
 [0.83333333 0.83333333 0.83333333 1.         1.        ]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
0.23.2
'''

import sklearn
print(sklearn.__version__) # 0.23.2

