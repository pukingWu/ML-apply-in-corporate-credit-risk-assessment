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
# 训练模型
svm_clf = SVC(kernel='rbf', probability=True,C=36.6811,gamma=11.5417 ,random_state=0)
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=16, min_samples_split=45, min_samples_leaf=5,random_state=0)
rf_clf = RandomForestClassifier(n_estimators=146, max_depth=24,min_samples_split=51,min_samples_leaf=14, random_state=0)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', eta=0.3282,gamma=0.0447,max_depth=17, n_estimators=277,random_state=0)
# 初始化逻辑回归模型，并设置L2正则化
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=23.1525, random_state=0)

# 训练模型
svm_clf.fit(X_resampled, y_resampled)
y_pred_svm = svm_clf.predict(X_test)
kappa_svm = cohen_kappa_score(y_test,y_pred_svm,weights='quadratic')

rf_clf.fit(X_resampled, y_resampled)
y_pred_rf = rf_clf.predict(X_test)
kappa_rf = cohen_kappa_score(y_test,y_pred_rf,weights='quadratic')

xgb_clf.fit(X_resampled, y_resampled)
y_pred_xgb = xgb_clf.predict(X_test)
kappa_xgb = cohen_kappa_score(y_test,y_pred_xgb,weights='quadratic')

clf.fit(X_resampled,y_resampled)
y_pred_clf = clf.predict(X_test)
kappa_clf = cohen_kappa_score(y_test,y_pred_clf,weights='quadratic')

dt_clf.fit(X_resampled,y_resampled)
y_pred_dt = dt_clf.predict(X_test)
kappa_dt = cohen_kappa_score(y_test,y_pred_dt,weights='quadratic')
 #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
# print("Kappa系数为：",kappa)
# 自定义集成策略（加权投票）
# global ensemble_probs
from sklearn.metrics import roc_auc_score
def weighted_voting(X):
    svm_probs = svm_clf.predict_proba(X)
    rf_probs = rf_clf.predict_proba(X)
    xgb_probs = xgb_clf.predict_proba(X)
    clf_probs = clf.predict_proba(X)
    dt_probs = dt_clf.predict_proba(X)
    # 假设我们为每个模型分配的权重
    # 14 2 1 7
    weights = [kappa_svm,kappa_rf,kappa_xgb,kappa_clf,kappa_dt]  # 根据模型性能调整权重
    # 加权集成预测概率
    ensemble_probs = np.average([svm_probs, rf_probs, xgb_probs,clf_probs,dt_probs], axis=0, weights=weights)
    # 将加权后的概率转换为类别预测（取最大概率的类别）
    # macro_roc_auc = roc_auc_score(y_test, ensemble_probs, multi_class='ovr', average='macro')  # 宏平均
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)
    return ensemble_predictions

# 获取集成预测
ensemble_predictions = weighted_voting(X_test)
# Lift for Class 0: 13.42
# Lift for Class 1: 2.99
# Lift for Class 2: 1.72
# Lift for Class 3: 1.44
# Lift for Class 4: 3.78
# Lift for Class 5: 25.88
import matplotlib.pyplot as plt
# 计算每个类别的真实和预测的正样本数量
n_samples_test = len(y_test)
class_counts = np.bincount(y_test)
class_prob_overall = class_counts / n_samples_test

# 初始化lift数组
lift_scores = []

# 计算每个类别的lift
for class_idx in range(len(class_prob_overall)):
    class_mask = y_test == class_idx
    class_samples = np.sum(class_mask)

    # 计算该类别的预测为正样本的数量
    predicted_positives = np.sum(ensemble_predictions[class_mask] == class_idx)

    # 计算该类别的响应率（真实正样本率）
    response_rate_targeted = predicted_positives / class_samples if class_samples > 0 else 0

    # 计算lift
    lift_score = response_rate_targeted / class_prob_overall[class_idx]
    lift_scores.append(lift_score)

# 绘制lift图
classes = np.arange(len(class_prob_overall))
plt.bar(classes, lift_scores)
# 添加数值标注

for idx, lift in enumerate(lift_scores):
    plt.text(idx, lift + 0.1,  # 稍微向上偏移一些，避免与柱子重叠
             f'{lift:.2f}',  # 保留两位小数
             ha='center')  # 水平居中对齐
plt.title('Lift Scores for Each Class')
plt.xlabel('Class Index')
plt.ylabel('Lift Score')
plt.xticks(classes, [f'Class {i}' for i in classes])
plt.show()

# 打印每个类别的lift分数
for class_idx, lift_score in enumerate(lift_scores):
    print(f"Lift for Class {class_idx}: {lift_score:.2f}")
# #获取预测概率
# print(ensemble_predictions)
# y_pred_proba = meta_model.predict_proba(S_test_flat)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
# from sklearn.metrics import roc_auc_score
# macro_roc_auc = roc_auc_score(y_test, ensemble_probs, multi_class='ovr', average='macro')  # 宏平均
# print("宏平均ROC AUC分数为:", macro_roc_auc)
# 评估准确率
# accuracy = accuracy_score(y_test, ensemble_predictions)
# print("Accuracy of the ensemble model:", accuracy)
#
# from sklearn.metrics import cohen_kappa_score
# kappa = cohen_kappa_score(y_test,ensemble_predictions,weights='quadratic') #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
# print("Kappa系数为：",kappa)
# #
# from sklearn.metrics import precision_score,recall_score,f1_score
# #计算宏查准率
# macro_precision = precision_score(y_test, ensemble_predictions, zero_division=1,average="macro")
# print("宏查准率为:",macro_precision)
# #计算宏查全率
# macro_recall = recall_score(y_test, ensemble_predictions, zero_division=1,average="macro")
# print("宏查全率为:",macro_recall)
# #计算f1指数
# macro_f1 = f1_score(y_test, ensemble_predictions, zero_division=1,average="macro")
# print("f1指数为：",macro_f1)

# 100%|██████████| 200/200 [04:57<00:00,  1.49s/trial, best loss: -0.4963768115942029]
# {'C': 34.768972159346916, 'C2': 30.82910322825633, 'eta': 0.34024331686505616, 'gamma': 19.76874486741419, 'gamma1': 0.01863643522695585, 'max_depth': 30.0, 'max_depth0': 5.731989133081535, 'max_depth1': 7.118303059401282, 'min_samples_leaf': 45.0, 'min_samples_leaf0': 18.500930048271748, 'min_samples_split': 36.0, 'min_samples_split0': 31.045348570255914, 'n_estimators': 162.96320612805982, 'n_estimators1': 189.3570405029336}
# Accuracy of the ensemble model: 0.48188405797101447
# Kappa系数为： 0.6160719062650474
# 宏查准率为: 0.443548501695788
# 宏查全率为: 0.5682244205573854
# f1指数为： 0.4669396923934614

# 100%|██████████| 200/200 [05:11<00:00,  1.56s/trial, best loss: -0.5]
# {'C': 24.869336649637017, 'C2': 26.569679786098703, 'eta': 0.25857480797872573, 'gamma': 17.979765209225704, 'gamma1': 0.009881890193571052, 'max_depth': 12.0, 'max_depth0': 2.939141245261335, 'max_depth1': 8.556357712397462, 'min_samples_leaf': 13.0, 'min_samples_leaf0': 8.76710280477812, 'min_samples_split': 74.0, 'min_samples_split0': 30.515505406223113, 'n_estimators': 105.57895726359703, 'n_estimators1': 125.91170394839808}
# Accuracy of the ensemble model: 0.4927536231884058
# Kappa系数为： 0.5821257550624395
# 宏查准率为: 0.4368695334717339
# 宏查全率为: 0.5165424541003942
# f1指数为： 0.45031560526952236

# 100%|██████████| 200/200 [05:28<00:00,  1.64s/trial, best loss: -0.5]
# {'C': 48.459163129217906, 'C2': 33.45604914272744, 'eta': 0.11457142574142792, 'gamma': 16.91785675016018, 'gamma1': 0.19564143225897293, 'max_depth': 7.0, 'max_depth0': 6.500367225815944, 'max_depth1': 7.610629060246433, 'min_samples_leaf': 12.0, 'min_samples_leaf0': 16.69623321968236, 'min_samples_split': 88.0, 'min_samples_split0': 40.019165591031665, 'n_estimators': 106.09488647155618, 'n_estimators1': 177.22664391050074}
# Accuracy of the ensemble model: 0.4855072463768116
# Kappa系数为： 0.6002390064592251
# 宏查准率为: 0.43962779495593285
# 宏查全率为: 0.5487640115669521
# f1指数为： 0.45967138055993284


# 100%|██████████| 200/200 [06:26<00:00,  1.93s/trial, best loss: -0.5072463768115942]
# {'C': 20.920039974251278, 'C2': 25.652509934467467, 'eta': 0.42635644357715485, 'gamma': 17.504171243675035, 'gamma1': 0.11600868827283801, 'max_depth': 30.0, 'max_depth0': 11.500674388392696, 'max_depth1': 10.436061183215404, 'min_samples_leaf': 11.0, 'min_samples_leaf0': 14.34784246174333, 'min_samples_split': 10.0, 'min_samples_split0': 49.97585382696815, 'n_estimators': 262.10312906767115, 'n_estimators1': 163.85199882015007}
# Accuracy of the ensemble model: 0.4891304347826087
# Kappa系数为： 0.5807203010213223
# 宏查准率为: 0.4406585389053832
# 宏查全率为: 0.534544200686612
# f1指数为： 0.45632850131477426
#
# 100%|██████████| 200/200 [05:53<00:00,  1.77s/trial, best loss: -0.5181159420289855]
# {'C': 38.560034654521836, 'C2': 30.738625041209893, 'eta': 0.038512052911642225, 'gamma': 7.047291684675567, 'gamma1': 0.17191826303730537, 'max_depth': 8.0, 'max_depth0': 10.023177561099692, 'max_depth1': 10.474532787406698, 'min_samples_leaf': 5.0, 'min_samples_leaf0': 3.0576858661262767, 'min_samples_split': 55.0, 'min_samples_split0': 42.71512992999689, 'n_estimators': 80.59239562341922, 'n_estimators1': 266.03004244360045}
# Accuracy of the ensemble model: 0.5072463768115942
# Kappa系数为： 0.5779849605194032
# 宏查准率为: 0.4499293630495676
# 宏查全率为: 0.531084236682509
# f1指数为： 0.4653983960768697
#将模型权重设为Kappa系数后
# Accuracy of the ensemble model: 0.5144927536231884
# Kappa系数为： 0.5908453104809621
# 宏查准率为: 0.45634119319369537
# 宏查全率为: 0.5362586468721168
# f1指数为： 0.4730045671248739


# 100%|██████████| 200/200 [05:04<00:00,  1.52s/trial, best loss: -0.5108695652173914]
# {'C': 32.612562806484256, 'C2': 24.319677500210453, 'eta': 0.43819172399095807, 'gamma': 6.033575732303907, 'gamma1': 0.09086145845533644, 'max_depth': 6.0, 'max_depth0': 15.183535346859555, 'max_depth1': 6.67404445759747, 'min_samples_leaf': 87.0, 'min_samples_leaf0': 27.823589558247072, 'min_samples_split': 76.0, 'min_samples_split0': 30.5758902680749, 'n_estimators': 299.80684101773056, 'n_estimators1': 288.33742450207296}
#Accuracy of the ensemble model: 0.4927536231884058
# Kappa系数为： 0.580338266384778
# 宏查准率为: 0.44159145361975544
# 宏查全率为: 0.5339643397134536
# f1指数为： 0.45920818272303426

# 100%|██████████| 200/200 [05:19<00:00,  1.60s/trial, best loss: -0.5108695652173914]
# {'C': 23.4173814590325, 'C2': 22.656203578668684, 'eta': 0.37651914629128047, 'gamma': 5.386456182643087, 'gamma1': 0.07862455457406758, 'max_depth': 30.0, 'max_depth0': 11.071105907124593, 'max_depth1': 11.013252684919145, 'min_samples_leaf': 117.0, 'min_samples_leaf0': 30.980270824947617, 'min_samples_split': 20.0, 'min_samples_split0': 29.783865156345538, 'n_estimators': 58.68916377740646, 'n_estimators1': 276.7230286799355}
# Accuracy of the ensemble model: 0.5108695652173914
# Kappa系数为： 0.5742930591259641
# 宏查准率为: 0.4563127413127413
# 宏查全率为: 0.5479464356745225
# f1指数为： 0.4736901640286051

# 100%|██████████| 200/200 [04:56<00:00,  1.48s/trial, best loss: -0.5072463768115942]
# {'C': 15.867390062713428, 'C2': 35.9807760588396, 'eta': 0.48717396237563704, 'gamma': 7.752557277133264, 'gamma1': 0.39533118180159343, 'max_depth': 5.0, 'max_depth0': 13.43074077320427, 'max_depth1': 7.032745557522729, 'min_samples_leaf': 113.0, 'min_samples_leaf0': 7.114955061544521, 'min_samples_split': 75.0, 'min_samples_split0': 31.02541131183203, 'n_estimators': 28.433536605905623, 'n_estimators1': 44.408282696066024}
# Accuracy of the ensemble model: 0.5072463768115942
# Kappa系数为： 0.5837293376685387
# 宏查准率为: 0.45804245380400704
# 宏查全率为: 0.5631741579691141
# f1指数为： 0.47586047642197377



# 100%|██████████| 200/200 [05:40<00:00,  1.70s/trial, best loss: -0.5144927536231884]
# {'C': 36.681157465956865, 'C2': 23.15254339477504, 'eta': 0.3282387191454794, 'gamma': 11.541731660576, 'gamma1': 0.044728718381354296, 'max_depth': 24.0, 'max_depth0': 16.487693851303007, 'max_depth1': 17.679982777895248, 'min_samples_leaf': 14.0, 'min_samples_leaf0': 5.63042350084045, 'min_samples_split': 51.0, 'min_samples_split0': 45.40929882701654, 'n_estimators': 146.9185220767554, 'n_estimators1': 277.2954242500145}
# Accuracy of the ensemble model: 0.5144927536231884
# Kappa系数为： 0.5934616272681672
# 宏查准率为: 0.46746074421878164
# 宏查全率为: 0.5669420851214737
# f1指数为： 0.49215174475898255



