from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
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
 [0.85714286 1.         1.         0.85714286 0.57142857]
BaggingClassifier 의 정답율 : 
 [1.         1.         0.85714286 1.         1.        ]
BernoulliNB 의 정답율 :
 [0.57142857 0.42857143 0.42857143 0.42857143 0.28571429]
CalibratedClassifierCV 의 정답율 : 
 [1.         1.         1.         0.57142857 1.        ]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답율 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답율 :
 [0.85714286 0.28571429 1.         1.         0.85714286]
DecisionTreeClassifier 의 정답율 :
 [1.         0.85714286 0.57142857 0.85714286 0.85714286]
DummyClassifier 의 정답율 :
 [0.28571429 0.28571429 0.57142857 0.42857143 0.42857143]
ExtraTreeClassifier 의 정답율 :
 [1.         1.         0.71428571 0.85714286 1.        ]
ExtraTreesClassifier 의 정답율 : 
 [1.         1.         0.85714286 1.         1.        ]
GaussianNB 의 정답율 :
 [1. 1. 1. 1. 1.]
GaussianProcessClassifier 의 정답율 : 
 [0.14285714 0.57142857 0.28571429 0.71428571 0.42857143]
GradientBoostingClassifier 의 정답율 : 
 [0.71428571 0.71428571 0.71428571 1.         0.85714286]
HistGradientBoostingClassifier 의 정답율 : 
 [0.28571429 0.57142857 0.28571429 0.42857143 0.57142857]
KNeighborsClassifier 의 정답율 :
 [0.85714286 0.42857143 0.71428571 0.71428571 0.57142857]
LabelPropagation 의 정답율 :
 [0.14285714 0.14285714 0.42857143 0.57142857 0.        ]
LabelSpreading 의 정답율 : 
 [0.28571429 0.28571429 0.14285714 0.14285714 0.42857143]
LinearDiscriminantAnalysis 의 정답율 :
 [0.85714286 0.85714286 0.57142857 1.         0.85714286]
LinearSVC 의 정답율 : 
 [0.42857143 0.57142857 0.85714286 0.71428571 0.42857143]
LogisticRegression 의 정답율 : 
 [0.85714286 1.         0.85714286 0.85714286 0.85714286]
LogisticRegressionCV 의 정답율 : 
 [1.         0.71428571 0.85714286 1.         1.        ]
MLPClassifier 의 정답율 : 
 [0.85714286 0.28571429 1.         0.85714286 1.        ]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답율 : 
 [1.         1.         1.         0.57142857 1.        ]
NearestCentroid 의 정답율 :
 [1.         0.71428571 0.71428571 0.57142857 0.42857143]
NuSVC 의 정답율 :
 [0.85714286 0.57142857 0.71428571 1.         0.71428571]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답율 : 
 [0.42857143 0.57142857 0.42857143 0.71428571 0.14285714]
Perceptron 의 정답율 : 
 [0.42857143 0.28571429 0.14285714 0.28571429 0.28571429]
QuadraticDiscriminantAnalysis 의 정답율 :
 [0.42857143 0.57142857 0.42857143 0.14285714 0.42857143]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답율 : 
 [1.         1.         1.         0.85714286 0.85714286]
RidgeClassifier 의 정답율 : 
 [1.         0.85714286 0.85714286 1.         0.85714286]
RidgeClassifierCV 의 정답율 :
 [0.85714286 1.         1.         1.         1.        ]
SGDClassifier 의 정답율 : 
 [0.57142857 0.28571429 0.57142857 0.14285714 0.57142857]
SVC 의 정답율 :
 [0.42857143 0.42857143 0.42857143 0.71428571 0.57142857]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
0.23.2
'''

import sklearn
print(sklearn.__version__) # 0.23.2

