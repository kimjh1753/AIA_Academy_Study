from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # shuffled은 행을 섞는다

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
 [0.39918164 0.7996264  0.76142522 0.5287756  0.56917074]
AdaBoostRegressor 의 정답율 : 
 [0.82355194 0.72480612 0.76880788 0.34440864 0.73326931]
BaggingRegressor 의 정답율 : 
 [0.84408615 0.7674376  0.91263361 0.74860117 0.65143937]
BayesianRidge 의 정답율 :
 [0.53295987 0.77448632 0.16069279 0.73336418 0.76271895]
CCA 의 정답율 : 
 [0.61076723 0.58883742 0.65107177 0.70105017 0.6936064 ]
DecisionTreeRegressor 의 정답율 :
 [ 0.87961742 -0.30699487  0.81519411  0.91530779  0.43476818]
DummyRegressor 의 정답율 :
 [-0.00590642 -0.04976731 -0.03170406 -0.00663564 -0.09407281]
ElasticNet 의 정답율 : 
 [0.77513947 0.41321086 0.65003895 0.74603161 0.7318816 ]
ElasticNetCV 의 정답율 : 
 [0.77381136 0.73391328 0.38537914 0.650956   0.63348648]
ExtraTreeRegressor 의 정답율 :
 [-0.47617763  0.82513454  0.77831634  0.6326443   0.47124488]
ExtraTreesRegressor 의 정답율 : 
 [0.83913779 0.82563852 0.91665995 0.72361747 0.76230352]
GammaRegressor 의 정답율 :
 [-0.00835035 -0.03653086 -0.00193849 -0.17887706 -0.00770321]
GaussianProcessRegressor 의 정답율 : 
 [-12.23420834  -6.40156623  -4.49999485  -8.92612054  -7.19416637]
GeneralizedLinearRegressor 의 정답율 : 
 [0.73300552 0.64467766 0.81488545 0.74645761 0.59793285]
GradientBoostingRegressor 의 정답율 : 
 [0.88469648 0.68967797 0.69857095 0.88043516 0.93646055]
HistGradientBoostingRegressor 의 정답율 : 
 [0.61856936 0.72529448 0.54871632 0.6932579  0.68705087]
HuberRegressor 의 정답율 : 
 [0.7154002  0.61375352 0.44435141 0.55346038 0.86352405]
IsotonicRegression 의 정답율 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답율 :
 [ 0.03927633 -0.39003362  0.23529552  0.37398214  0.67818613]
KernelRidge 의 정답율 :
 [0.75054332 0.84277241 0.66123436 0.68704201 0.66663163]
Lars 의 정답율 : 
 [ 0.80467673  0.51960485  0.79709254  0.72538151 -0.11760369]
LarsCV 의 정답율 : 
 [0.69328111 0.69191515 0.82196669 0.65373011 0.73390283]
Lasso 의 정답율 :
 [0.6056127  0.85271729 0.59219141 0.65228021 0.20278904]
LassoCV 의 정답율 : 
 [0.81764718 0.74739909 0.75480678 0.77402824 0.61129514]
LassoLars 의 정답율 :
 [-0.02898456 -0.00046499 -0.25904604 -0.02299163 -0.33656611]
LassoLarsCV 의 정답율 : 
 [0.82407834 0.77618593 0.76837156 0.23612105 0.45773036]
LassoLarsIC 의 정답율 :
 [0.59970125 0.77992884 0.8071737  0.81125005 0.6527991 ]
LinearRegression 의 정답율 : 
 [0.77978156 0.75792765 0.79497913 0.63606218 0.8089414 ]
LinearSVR 의 정답율 : 
 [-0.60789472 -0.1321774   0.31407987  0.22479761  0.67249691]
MLPRegressor 의 정답율 : 
 [  0.04624317 -15.53905891  -9.19168908   0.56943815   0.4110313 ]
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
 [ 0.13278747  0.32832138  0.33470353  0.3911771  -0.09660599]
OrthogonalMatchingPursuit 의 정답율 : 
 [0.47700634 0.25173745 0.33583682 0.3010769  0.43088116]
OrthogonalMatchingPursuitCV 의 정답율 : 
 [0.76920987 0.75222953 0.75035683 0.45203105 0.66624242]
PLSCanonical 의 정답율 : 
 [-5.71139275  0.09972483 -6.57627359 -8.48356735 -4.00066421]
PLSRegression 의 정답율 :
 [0.61050707 0.50502974 0.79225737 0.77343596 0.78850266]
PassiveAggressiveRegressor 의 정답율 :
 [ -0.0225987   -0.71044578 -19.74496403  -0.2626062   -0.28078823]
PoissonRegressor 의 정답율 : 
 [0.84919963 0.84402309 0.74820782 0.81561519 0.78191886]
RANSACRegressor 의 정답율 : 
 [ 0.67470856  0.69883619 -0.99964426  0.03504932  0.56754163]
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답율 : 
 [0.83509059 0.84276587 0.88243897 0.66661257 0.82650072]
RegressorChain 은 없는 놈!
Ridge 의 정답율 :
 [0.84526527 0.70961083 0.72528874 0.68393534 0.73244849]
RidgeCV 의 정답율 : 
 [0.77205459 0.65920291 0.7007557  0.72537601 0.63302563]
SGDRegressor 의 정답율 :
 [-3.82373384e+27 -4.32317628e+26 -4.49857597e+25 -3.57199691e+27
 -1.03995879e+27]
SVR 의 정답율 : 
 [0.21684341 0.02579048 0.20136198 0.39140629 0.14391011]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답율 : 
 [0.75052566 0.75087975 0.33949049 0.68606453 0.82711818]
TransformedTargetRegressor 의 정답율 :
 [0.73298874 0.51112105 0.8260142  0.88309807 0.69961589]
TweedieRegressor 의 정답율 : 
 [0.43999586 0.67700166 0.5687983  0.70203002 0.81270453]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답율 :
 [nan nan nan nan nan]
0.23.
'''

import sklearn
print(sklearn.__version__) # 0.23.2

