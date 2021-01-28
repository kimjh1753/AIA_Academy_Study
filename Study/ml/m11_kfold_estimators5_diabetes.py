from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # shuffle은 행을 섞는다

allAlgorithms = all_estimators(type_filter='regressor')

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
ARDRegression 의 정답율 : 
 [ 0.47626428  0.64701598 -0.36765233  0.26941695  0.34506181]
AdaBoostRegressor 의 정답율 : 
 [-0.10143905  0.51839221 -0.1496307   0.26211706  0.29153511]
BaggingRegressor 의 정답율 : 
 [-0.11258906  0.04188441  0.4164649  -0.21175786  0.2865178 ]
BayesianRidge 의 정답율 :
 [0.29241333 0.5730019  0.21325405 0.30077531 0.3622313 ]
CCA 의 정답율 : 
 [-0.26669361  0.3800744   0.21883265  0.37943519  0.20371493]
DecisionTreeRegressor 의 정답율 :
 [ 0.08133457 -0.19850475 -0.93136105 -2.37367599 -0.02371848]
DummyRegressor 의 정답율 : 
 [-0.01836309 -0.08001727 -0.00910605 -0.00141735 -0.04557755]
ElasticNet 의 정답율 :
 [-0.00820231 -0.01600275 -0.03261481 -0.00598705  0.00759202]
ElasticNetCV 의 정답율 : 
 [0.25356403 0.17764375 0.30232728 0.38336056 0.37564016]
ExtraTreeRegressor 의 정답율 :
 [-0.11477276 -0.41335832 -1.60278793 -0.37900684  0.2632783 ]
ExtraTreesRegressor 의 정답율 : 
 [-0.3355926   0.39352834 -0.0315408   0.55309229  0.31829293]
GammaRegressor 의 정답율 :
 [-0.00805637 -0.01931911 -0.02240116 -0.2170045   0.00557885]
GaussianProcessRegressor 의 정답율 : 
 [-23.31481998  -6.8822809   -2.72747448 -10.0324933   -6.08971098]
GeneralizedLinearRegressor 의 정답율 :
 [-0.16710744  0.00392416 -0.01879287 -0.06226952  0.00222792]
GradientBoostingRegressor 의 정답율 : 
 [ 0.20991041 -0.70735193 -0.06600524  0.58808125  0.16054756]
HistGradientBoostingRegressor 의 정답율 : 
 [ 0.40906141  0.42294458  0.28668813 -0.54466759  0.25370901]
HuberRegressor 의 정답율 : 
 [0.32934349 0.27846315 0.25889226 0.08533139 0.46691847]
IsotonicRegression 의 정답율 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답율 : 
 [ 0.24802226 -0.24098998  0.09577288  0.39422181  0.34226221]
KernelRidge 의 정답율 :
 [-4.17342698 -3.25096212 -4.25439673 -3.16672256 -3.36041022]
Lars 의 정답율 : 
 [-7.16549673e+00  4.79478111e-01  2.32164959e-01 -2.69637846e-03
  1.36674183e-01]
LarsCV 의 정답율 : 
 [0.25496049 0.33871493 0.50797379 0.38487805 0.2215334 ]
Lasso 의 정답율 :
 [ 0.26997864 -0.31656625 -0.36653659  0.29870965  0.15776826]
LassoCV 의 정답율 : 
 [ 0.32459694  0.49593782  0.3948728  -0.28984385  0.23099285]
LassoLars 의 정답율 :
 [ 0.2760546   0.23792398  0.14536725 -0.84151574  0.45358333]
LassoLarsCV 의 정답율 : 
 [0.40489048 0.19331299 0.3659059  0.23925178 0.43378902]
LassoLarsIC 의 정답율 :
 [0.2215624  0.14127379 0.33959849 0.42869074 0.31100052]
LinearRegression 의 정답율 :
 [ 0.67726113  0.46858327  0.51697844 -0.49402761 -0.51821704]
LinearSVR 의 정답율 : 
 [-1.28326328 -0.98115802 -1.57537105 -1.34187245 -0.89314788]
MLPRegressor 의 정답율 : 
 [-3.01962531 -2.77591263 -3.58884364 -4.39911665 -3.04193664]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답율 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답율 :
 [nan nan nan nan nan]
MultiTaskLasso 의 정답율 :
 [nan nan nan nan nan]
MultiTaskLassoCV 의 정답율 : 
 [nan nan nan nan nan]
NuSVR 의 정답율 :
 [-0.03887818 -0.06045556  0.0387752   0.0398294  -0.66466707]
OrthogonalMatchingPursuit 의 정답율 :
 [0.04964263 0.34037654 0.24290734 0.35967561 0.17568267]
OrthogonalMatchingPursuitCV 의 정답율 : 
 [ 0.55589603  0.53502844  0.18759358 -0.36692756  0.4100397 ]
PLSCanonical 의 정답율 :
 [-4.16985236 -0.3238477  -4.04505529 -0.70412123 -3.31426775]
PLSRegression 의 정답율 :
 [ 0.51366816  0.19000771 -0.38047926  0.46421603  0.28109837]
PassiveAggressiveRegressor 의 정답율 :
 [ 0.27650022  0.10174712 -0.25462741  0.09153875  0.0664862 ]
PoissonRegressor 의 정답율 : 
 [ 0.31724241 -0.0122858   0.22693925  0.31338585  0.23144777]
RANSACRegressor 의 정답율 : 
 [-1.66206149 -0.3422428  -0.97161685  0.15808282  0.19739356]
RadiusNeighborsRegressor 의 정답율 :
 [-0.00290062 -0.09651425 -0.00351947 -0.20292181 -0.30029439]
RandomForestRegressor 의 정답율 : 
 [ 0.4145832   0.30218148 -0.00920062 -0.01876794  0.49172737]
RegressorChain 은 없는 놈!
Ridge 의 정답율 :
 [0.13245865 0.19793019 0.09849674 0.09074648 0.18074859]
RidgeCV 의 정답율 :
 [ 0.20900733  0.34568952  0.44762709  0.32554526 -0.22210877]
SGDRegressor 의 정답율 : 
 [0.07421193 0.2153382  0.22766269 0.04351259 0.24841239]
SVR 의 정답율 :
 [ 0.04153294 -0.34078492 -0.40042443 -0.14305731 -0.07859751]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답율 : 
 [ 0.66343307  0.2282013   0.35282601 -0.54924992  0.08876623]
TransformedTargetRegressor 의 정답율 :
 [ 0.67783795  0.19061578  0.23682895  0.58732578 -0.56010852]
TweedieRegressor 의 정답율 : 
 [-0.11408531 -0.00277569 -0.0142437   0.00374857 -0.00191947]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답율 :
 [nan nan nan nan nan]
0.23.2
'''

import sklearn
print(sklearn.__version__) # 0.23.2

