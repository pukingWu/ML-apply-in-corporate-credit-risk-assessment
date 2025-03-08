from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score

file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
# 注意：您应该使用 inplace=True 或重新赋值来确保替换是永久性的
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)
X = cred.drop(columns=['企业信用等级'])
y = cred['企业信用等级'].astype(int)
# 划分数据集为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
# 设置模型参数
svm_clf = SVC(kernel='rbf', probability=True, random_state=0)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=64.9387, random_state=0)
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=8, min_samples_leaf=6,random_state=0 )
# 训练模型
svm_clf.fit(X_resampled, y_resampled)
rf_clf.fit(X_resampled, y_resampled)
xgb_clf.fit(X_resampled, y_resampled)
clf.fit(X_resampled,y_resampled)
dt_clf.fit(X_resampled,y_resampled)
# 自定义集成策略（加权投票）
def weighted_voting(X):
    svm_probs = svm_clf.predict_proba(X)
    rf_probs = rf_clf.predict_proba(X)
    xgb_probs = xgb_clf.predict_proba(X)
    clf_probs = clf.predict_proba(X)
    dt_probs = dt_clf.predict_proba(X)
    # 假设我们为每个模型分配的权重
    weights = [0.2, 0.2, 0.2,0.2,0.2]  # 根据模型性能调整权重
    # 加权集成预测概率
    ensemble_probs = np.average([svm_probs, rf_probs, xgb_probs,clf_probs,dt_probs], axis=0, weights=weights)
    # 将加权后的概率转换为类别预测（取最大概率的类别）
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)
    return ensemble_predictions

# 获取集成预测
ensemble_predictions = weighted_voting(X_test)
# 评估准确率
accuracy = accuracy_score(y_test, ensemble_predictions)
print("Accuracy of the ensemble model:", accuracy)

def percept(args):

    # 设置模型参数
    svm_clf = SVC(probability=True, kernel='rbf', C=args["C"], gamma=args["gamma"], random_state=0)
    dt_clf= DecisionTreeClassifier(criterion='entropy', max_depth=int(args["max_depth0"]),min_samples_split=int(args["min_samples_split0"]),min_samples_leaf=int(args["min_samples_leaf0"]),random_state=0)
    rf_clf = RandomForestClassifier(n_estimators=int(args["n_estimators"]), max_depth=int(args["max_depth"]),min_samples_split=int(args["min_samples_split"]),min_samples_leaf=int(args["min_samples_leaf"]),random_state=0)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=int(args["max_depth1"]),eta=float(args["eta"]), gamma=float(args["gamma1"]),n_estimators=int(args["n_estimators1"]), random_state=0)
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=float(args["C2"]), random_state=0)

    # 训练模型
    svm_clf.fit(X_resampled, y_resampled)
    y_pred_svm = svm_clf.predict(X_test)
    kappa_svm = cohen_kappa_score(y_test, y_pred_svm, weights='quadratic')

    rf_clf.fit(X_resampled, y_resampled)
    y_pred_rf = rf_clf.predict(X_test)
    kappa_rf = cohen_kappa_score(y_test, y_pred_rf, weights='quadratic')

    xgb_clf.fit(X_resampled, y_resampled)
    y_pred_xgb = xgb_clf.predict(X_test)
    kappa_xgb = cohen_kappa_score(y_test, y_pred_xgb, weights='quadratic')

    clf.fit(X_resampled, y_resampled)
    y_pred_clf = clf.predict(X_test)
    kappa_clf = cohen_kappa_score(y_test, y_pred_clf, weights='quadratic')

    dt_clf.fit(X_resampled, y_resampled)
    y_pred_dt = dt_clf.predict(X_test)
    kappa_dt = cohen_kappa_score(y_test, y_pred_dt, weights='quadratic')

    # (label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
    # print("Kappa系数为：",kappa)
    # 自定义集成策略（加权投票）
    def weighted_voting(X):
        svm_probs = svm_clf.predict_proba(X)
        rf_probs = rf_clf.predict_proba(X)
        xgb_probs = xgb_clf.predict_proba(X)
        clf_probs = clf.predict_proba(X)
        dt_probs = dt_clf.predict_proba(X)
        # 假设我们为每个模型分配的权重
        # 14 2 1 7
        weights = [kappa_svm, kappa_rf, kappa_xgb, kappa_clf, kappa_dt]  # 根据模型性能调整权重
        # 加权集成预测概率
        ensemble_probs = np.average([svm_probs, rf_probs, xgb_probs, clf_probs, dt_probs], axis=0, weights=weights)
        # 将加权后的概率转换为类别预测（取最大概率的类别）
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        return ensemble_predictions

    # 获取集成预测
    ensemble_predictions = weighted_voting(X_test)
    # 评估准确率
    # accuracy = accuracy_score(y_test, ensemble_predictions)
    return -accuracy_score(y_test, ensemble_predictions)

from hyperopt import fmin,tpe,hp,partial

space = {#SVM参数
         "C":hp.uniform("C",0.01,50),
         "gamma":hp.uniform("gamma",0.01,20),
    # 决策树参数
         "max_depth0": hp.uniform("max_depth0", 2, 20),
         "min_samples_split0": hp.uniform("min_samples_split0", 2, 50),
         "min_samples_leaf0": hp.uniform("min_samples_leaf0", 2, 50),
    #随机森林参数
         "n_estimators":hp.uniform("n_estimators",2,300),
         "max_depth":hp.quniform("max_depth",2,35,1),
         "min_samples_split":hp.quniform("min_samples_split",2,121,1),
         "min_samples_leaf":hp.quniform("min_samples_leaf",2,121,1),
    #XGBoost参数
         "eta":hp.uniform("eta",0.01,0.5),
         "max_depth1":hp.uniform("max_depth1",2,20),
         "gamma1":hp.uniform("gamma1",0.001,1),
         "n_estimators1":hp.uniform("n_estimators1",2,301),
    #softmax参数
         "C2":hp.uniform("C2",0.01,50)
         }
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))

# 100%|██████████| 200/200 [04:57<00:00,  1.49s/trial, best loss: -0.4963768115942029]
# {'C': 34.768972159346916, 'C2': 30.82910322825633, 'eta': 0.34024331686505616, 'gamma': 19.76874486741419, 'gamma1': 0.01863643522695585, 'max_depth': 30.0, 'max_depth0': 5.731989133081535, 'max_depth1': 7.118303059401282, 'min_samples_leaf': 45.0, 'min_samples_leaf0': 18.500930048271748, 'min_samples_split': 36.0, 'min_samples_split0': 31.045348570255914, 'n_estimators': 162.96320612805982, 'n_estimators1': 189.3570405029336}


