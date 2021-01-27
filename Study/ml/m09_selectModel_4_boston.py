from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답을 : ', r2_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

'''        
  warnings.warn(message, FutureWarning)
ARDRegression 의 정답을 :  0.6734031740560393
AdaBoostRegressor 의 정답을 :  0.7235808949402059
BaggingRegressor 의 정답을 :  0.7560076580581017
BayesianRidge 의 정답을 :  0.6769129535307645
CCA 의 정답을 :  0.6298231660988343
DecisionTreeRegressor 의 정답을 :  0.6695958188339348
DummyRegressor 의 정답을 :  -0.05367349153667189
ElasticNet 의 정답을 :  0.6429155738893424
ElasticNetCV 의 정답을 :  0.6284285478132856
ExtraTreeRegressor 의 정답을 :  0.5534732455574753
ExtraTreesRegressor 의 정답을 :  0.775929002676891
GammaRegressor 의 정답을 :  -0.05367349153667189
GaussianProcessRegressor 의 정답을 :  -5.937496107900878
GeneralizedLinearRegressor 의 정답을 :  0.5769751632226707
GradientBoostingRegressor 의 정답을 :  0.7513604961921854
HistGradientBoostingRegressor 의 정답을 :  0.7023590846066745
HuberRegressor 의 정답을 :  0.47629190797457055
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답을 :  0.2472066141336704
KernelRidge 의 정답을 :  0.6680857835376558
Lars 의 정답을 :  0.6869519021019594
LarsCV 의 정답을 :  0.6726340221509277
Lasso 의 정답을 :  0.646176343885901
LassoCV 의 정답을 :  0.6653866064553682
LassoLars 의 정답을 :  -0.05367349153667189
LassoLarsCV 의 정답을 :  0.6726340221509277
LassoLarsIC 의 정답을 :  0.6343735014174723
LinearRegression 의 정답을 :  0.6869519021019591
LinearSVR 의 정답을 :  0.20369694809537287
MLPRegressor 의 정답을 :  0.3410116187911013
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답을 :  0.07054035330254693
OrthogonalMatchingPursuit 의 정답을 :  0.5113458676475233
OrthogonalMatchingPursuitCV 의 정답을 :  0.6660205513283118
PLSCanonical 의 정답을 :  -1.6071467819822858
PLSRegression 의 정답을 :  0.6766573841383329
PassiveAggressiveRegressor 의 정답을 :  -0.019421994812896415
PoissonRegressor 의 정답을 :  0.7304564151143369
RANSACRegressor 의 정답을 :  0.39787244156062895
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답을 :  0.7445310141522627
RegressorChain 은 없는 놈!
Ridge 의 정답을 :  0.6825519690752229
RidgeCV 의 정답을 :  0.682551969100837
SGDRegressor 의 정답을 :  -1.599606896254521e+26
SVR 의 정답을 :  0.05272477113409446
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답을 :  0.6262532641096135
TransformedTargetRegressor 의 정답을 :  0.6869519021019591
TweedieRegressor 의 정답을 :  0.5769751632226707
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''

import sklearn
print(sklearn.__version__) # 0.23.2