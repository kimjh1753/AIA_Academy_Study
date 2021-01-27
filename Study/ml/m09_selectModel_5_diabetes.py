from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답율 : ', r2_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

'''        
  warnings.warn(message, FutureWarning)
ARDRegression 의 정답율 :  0.42808163633835694
AdaBoostRegressor 의 정답율 :  0.34355640975574975
BaggingRegressor 의 정답율 :  0.37274945309576046 
BayesianRidge 의 정답율 :  0.41939075997350184    
CCA 의 정답율 :  0.41591448412804866
DecisionTreeRegressor 의 정답율 :  -0.12729253680875074
DummyRegressor 의 정답율 :  -0.13255315075218932       
ElasticNet 의 정답율 :  -0.1254824301041535
ElasticNetCV 의 정답율 :  0.3317483058612507        
ExtraTreeRegressor 의 정답율 :  -0.07127063057006011
ExtraTreesRegressor 의 정답율 :  0.34638002257098843
GammaRegressor 의 정답율 :  -0.12775908547295045
GaussianProcessRegressor 의 정답율 :  -6.898919978282699
GeneralizedLinearRegressor 의 정답율 :  -0.12669232385103024
GradientBoostingRegressor 의 정답율 :  0.2993607852654109
HistGradientBoostingRegressor 의 정답율 :  0.3063031091578622
HuberRegressor 의 정답율 :  0.38604157687202534
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답율 :  0.2829481609732153
KernelRidge 의 정답율 :  -4.050574116718863
Lars 의 정답율 :  0.411332922195782
LarsCV 의 정답율 :  0.3643151462051928
Lasso 의 정답율 :  0.18878545325836948
LassoCV 의 정답율 :  0.3910767359574481
LassoLars 의 정답율 :  0.3839072399549758
LassoLarsCV 의 정답율 :  0.3930113334786094
LassoLarsIC 의 정답율 :  0.39815562861823617
LinearRegression 의 정답율 :  0.41133292219578355
LinearSVR 의 정답율 :  -1.4527917269691262
MLPRegressor 의 정답율 :  -3.908987603163668
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답율 :  -0.12323372698775326
OrthogonalMatchingPursuit 의 정답율 :  0.26191020903042295
OrthogonalMatchingPursuitCV 의 정답율 :  0.43120630104982904
PLSCanonical 의 정답율 :  -1.134774178851159
PLSRegression 의 정답율 :  0.4278023646695678
PassiveAggressiveRegressor 의 정답율 :  -0.0891921319219402
PoissonRegressor 의 정답율 :  0.18186430250675678
RANSACRegressor 의 정답율 :  -0.2466888432489862
RadiusNeighborsRegressor 의 정답율 :  -0.13255315075218932
RandomForestRegressor 의 정답율 :  0.34520039763967036
RegressorChain 은 없는 놈!
Ridge 의 정답율 :  0.12174957415051191
RidgeCV 의 정답율 :  0.37175408167497614
SGDRegressor 의 정답율 :  0.11625343907048424
SVR 의 정답율 :  -0.3782764029649124
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답율 :  0.29816166629516283
TransformedTargetRegressor 의 정답율 :  0.41133292219578355
TweedieRegressor 의 정답율 :  -0.12669232385103024
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''

# diabetes의 Regressor중에서 OrthogonalMatchingPursuitCV 의 정답율 :  0.43120630104982904이 제일 좋다.

import sklearn
print(sklearn.__version__) # 0.23.2