import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek

start = time.time()  # Python3.8不支持clock了，使用timer.perf_counter()
file = r'C:\Users\吴振波\Desktop\实验数据集_OpenMLselect3.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 划分训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

#模型初始化
#线性核
linear_svm = SVC(kernel='poly', C=7.8, random_state=0)
linear_svm.fit(X_resampled, y_resampled)

#
# 基于RBF非线性SVM模型训练
rbf_svm = SVC(kernel='rbf', C=7.8, gamma=0.6, random_state=0)
rbf_svm.fit(X_resampled, y_resampled)

#模型预测
linear_y_pred = linear_svm.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_y_pred)
#
rbf_y_pred = rbf_svm.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_y_pred)

print('分类准确度accuracy: ', linear_accuracy)
print('分类准确度accuracy: ', rbf_accuracy)

#模型参数优化
def percept(args):
    global X_resampled,y_resampled,y_test
    model = SVC(kernel='rbf', C=float(args["C"]), gamma=float(args["gamma"]), random_state=0)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial

space = {
    # "kernel":hp.choice("kernel",['rbf','poly','linear']),
         "C":hp.uniform("C",0.01,50),
         "gamma":hp.uniform("gamma",0.01,20)
         }
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))