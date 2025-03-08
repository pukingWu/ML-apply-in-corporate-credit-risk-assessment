import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)

X = cred.drop(columns=['企业信用等级'])
y = cred['企业信用等级'].astype(int)

# 编码标签为整数（如果它们还不是）
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y.astype(str))
# 假设我们要进行5折分层交叉验证

# skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
# print(y.astype())
# 一级模型列表
base_models = [
    ('svm', SVC(probability=True, kernel='rbf', C=14, gamma=10.5339, random_state=0)),
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=8, min_samples_leaf=19,random_state=0, )),
    ('rf',RandomForestClassifier(n_estimators=123, max_depth=30, min_samples_split=24, min_samples_leaf=29, random_state=0)),
    ('xgb', XGBClassifier(use_label_encoder=False,eval_metric='mlogloss', max_depth=10,eta=0.2799,gamma=0.5222,n_estimators=54,random_state=0))
]
#{'C': 14.669381247353652, 'eta': 0.27993403808703815, 'gamma': 10.533925113903367, 'gamma1': 0.5222687378816802, 'max_depth': 30.0, 'max_depth0': 4.291512789348619, 'max_depth1': 10.808300039116787, 'min_samples_leaf': 29.0, 'min_samples_leaf0': 19.01868782831434, 'min_samples_split': 24.0, 'min_samples_split0': 8.843342857832097, 'n_estimators': 123.09090691817615, 'n_estimators1': 54.017120287450936}
# Lift for Class 0: 3.83
# Lift for Class 1: 2.87
# Lift for Class 2: 2.40
# Lift for Class 3: 1.87
# Lift for Class 4: 3.13
# Lift for Class 5: 17.25
# 训练一级模型并获取预测概率
S_train = np.zeros((X_train.shape[0], len(base_models), 6))
for i, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)
    S_train[:, i, :] = model.predict_proba(X_train)  # 直接获取所有9个类别的概率

# 对于测试集，也需要做同样的处理
S_test = np.zeros((X_test.shape[0], len(base_models), 6))
for i, (name, model) in enumerate(base_models):
    S_test[:, i, :] = model.predict_proba(X_test)

# 扁平化特征
S_train_flat = S_train.reshape(S_train.shape[0], -1)
S_test_flat = S_test.reshape(S_test.shape[0], -1)

# 使用逻辑回归作为元模型
meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',penalty='l2', C=1, random_state=0)

# 拟合元模型
meta_model.fit(S_train_flat, y_train)

# 获取元模型的系数和截距项
meta_estimator = meta_model.final_estimator_   # 获取元模型，即LogisticRegression对象
# 输出系数

coef = meta_estimator.coef_
print("Coefficients:")
# co = pd.DataFrame(coef)
for i, coef_i in enumerate(coef):
    print(f"Class {i}: {coef_i}")
# 输出截距项
intercept = meta_estimator.intercept_
print("Intercept:")
print(intercept)

# 预测测试集
y_pred = meta_model.predict(S_test_flat)
#获取预测概率
y_pred_proba = meta_model.predict_proba(S_test_flat)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)
# 评估性能
print("Stacking Model Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

# from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic') #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print("Kappa系数为：",kappa)
#
# from sklearn.metrics import precision_score,recall_score,f1_score
#计算宏查准率
macro_precision = precision_score(y_test, y_pred, zero_division=1,average="macro")
print("宏查准率为:",macro_precision)
#计算宏查全率
macro_recall = recall_score(y_test, y_pred, zero_division=1,average="macro")
print("宏查全率为:",macro_recall)
#计算f1指数
macro_f1 = f1_score(y_test, y_pred, zero_division=1,average="macro")
print("f1指数为：",macro_f1)

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
    predicted_positives = np.sum(y_pred[class_mask] == class_idx)

    # 计算该类别的响应率（真实正样本率）
    response_rate_targeted = predicted_positives / class_samples if class_samples > 0 else 0

    # 计算lift
    lift_score = response_rate_targeted / class_prob_overall[class_idx]
    lift_scores.append(lift_score)

# 绘制lift图
classes = np.arange(len(class_prob_overall))
plt.bar(classes, lift_scores)
plt.title('Lift Scores for Each Class')
plt.xlabel('Class Index')
plt.ylabel('Lift Score')
plt.xticks(classes, [f'Class {i}' for i in classes])
plt.show()

# 打印每个类别的lift分数
for class_idx, lift_score in enumerate(lift_scores):
    print(f"Lift for Class {class_idx}: {lift_score:.2f}")
# ('svm', SVC(probability=True, kernel='rbf', C=31.1981, gamma=17.4871, random_state=0)),
# ('dt',
#  DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=12, min_samples_leaf=9, random_state=0, )),
# ('rf',
#  RandomForestClassifier(n_estimators=66, max_depth=16, min_samples_split=90, min_samples_leaf=115, random_state=0)),
# ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=17, eta=0.2504, gamma=0.2354,
#                       n_estimators=281, random_state=0))

# Stacking Model Accuracy: 0.54
# Kappa系数为： 0.566519705179551
# 宏查准率为: 0.6112157196409194
# 宏查全率为: 0.4549158333074257
# f1指数为： 0.4897402360456548

#{'C': 15.947032431909541, 'eta': 0.28150958439970614, 'gamma': 10.953943491158942, 'gamma1': 0.04761058335290208, 'max_depth': 11.0, 'max_depth0': 6.014643928222784, 'max_depth1': 14.89046036390142, 'min_samples_leaf': 61.0, 'min_samples_leaf0': 5.644981388481787, 'min_samples_split': 71.0, 'min_samples_split0': 8.401819394451547, 'n_estimators': 155.3886704191565, 'n_estimators1': 95.59999083315412}
# Stacking Model Accuracy: 0.57
# Kappa系数为： 0.6041992796856014
# 宏查准率为: 0.6755796594732169
# 宏查全率为: 0.5022346568240562
# f1指数为： 0.5507697963673918



# best loss: -0.5579710144927537]
#{'C': 14.669381247353652, 'eta': 0.27993403808703815, 'gamma': 10.533925113903367, 'gamma1': 0.5222687378816802, 'max_depth': 30.0, 'max_depth0': 4.291512789348619, 'max_depth1': 10.808300039116787, 'min_samples_leaf': 29.0, 'min_samples_leaf0': 19.01868782831434, 'min_samples_split': 24.0, 'min_samples_split0': 8.843342857832097, 'n_estimators': 123.09090691817615, 'n_estimators1': 54.017120287450936}
# Stacking Model Accuracy: 0.56
# Kappa系数为： 0.6199322064136301
# 宏查准率为: 0.7119297375700852
# 宏查全率为: 0.48649955055222227
# f1指数为： 0.5313452844450149

# {'C': 10.141195194487349, 'eta': 0.3168739828538865, 'gamma': 18.65703926915401, 'gamma1': 0.30685404355574103, 'max_depth': 9.0, 'max_depth0': 5.1800509699416, 'max_depth1': 11.00061306901458, 'min_samples_leaf': 19.0, 'min_samples_leaf0': 7.0677252391632575, 'min_samples_split': 58.0, 'min_samples_split0': 9.739773726285602, 'n_estimators': 186.5629172511468, 'n_estimators1': 211.2761616227268}
# -0.5507246376811594
# Stacking Model Accuracy: 0.56
# Kappa系数为： 0.591435952538808
# 宏查准率为: 0.715513771491001
# 宏查全率为: 0.48188547467624526
# f1指数为： 0.5287652695486796

# {'C': 45.985522151983844, 'eta': 0.1382550868682706, 'gamma': 19.930237327587832, 'gamma1': 0.05699431664851433, 'max_depth': 24.0, 'max_depth0': 6.576317846930055, 'max_depth1': 17.665981799206577, 'min_samples_leaf': 27.0, 'min_samples_leaf0': 5.0827871574455505, 'min_samples_split': 47.0, 'min_samples_split0': 12.701164817239764, 'n_estimators': 161.6710418923521, 'n_estimators1': 121.78527601354175}
# -0.5507246376811594
# Stacking Model Accuracy: 0.53
# Kappa系数为： 0.5972112080359503
# 宏查准率为: 0.6365540913335032
# 宏查全率为: 0.4629984066840685
# f1指数为： 0.5071454476665886


#数据不平衡处理后
# {'C': 30.962066005657853, 'eta': 0.2213785178673142, 'gamma': 18.897016630535827, 'gamma1': 0.049382423817955115, 'max_depth': 19.0, 'max_depth0': 3.448978588905521, 'max_depth1': 14.651669323901459, 'min_samples_leaf': 3.0, 'min_samples_leaf0': 6.962584541588166, 'min_samples_split': 68.0, 'min_samples_split0': 15.61853719252669, 'n_estimators': 126.76119889003365, 'n_estimators1': 145.09704304111648}
# -0.5434782608695652
# Stacking Model Accuracy: 0.52
# Kappa系数为： 0.5711785690247022
# 宏查准率为: 0.604282515758597
# 宏查全率为: 0.4446947277307009
# f1指数为： 0.481537619468654

# {'C': 30.867782121743133, 'eta': 0.1703324548596819, 'gamma': 19.32282230350437, 'gamma1': 0.029125234904974123, 'max_depth': 27.0, 'max_depth0': 5.214753131571717, 'max_depth1': 12.647188857462954, 'min_samples_leaf': 18.0, 'min_samples_leaf0': 4.007864838225531, 'min_samples_split': 103.0, 'min_samples_split0': 7.3107668191347575, 'n_estimators': 53.06731005515358, 'n_estimators1': 206.7589177625473}
# -0.5543478260869565
# Stacking Model Accuracy: 0.53
# Kappa系数为： 0.5917521661089813
# 宏查准率为: 0.6383799614534591
# 宏查全率为: 0.4663226781132692
# f1指数为： 0.5101909545208515

#改了参数范围
#******************
#{'C': 5.460899325155317, 'eta': 0.1361018645948583, 'gamma': 15.979995271784736, 'gamma1': 0.11682694136386962, 'max_depth': 8.0, 'max_depth0': 4.181083090003661, 'max_depth1': 10.480742249625672, 'min_samples_leaf': 10.0, 'min_samples_leaf0': 27.439485619748215, 'min_samples_split': 37.0, 'min_samples_split0': 48.35167011886939, 'n_estimators': 34.17932699399256, 'n_estimators1': 207.80214323170898}
# -0.5652173913043478
# Stacking Model Accuracy: 0.57
# Kappa系数为： 0.6215511186916826
# 宏查准率为: 0.6624679586704904
# 宏查全率为: 0.4909594481981654
# f1指数为： 0.5345847496246682

# {'C': 42.79477410721951, 'eta': 0.07542852041839641, 'gamma': 17.727397495488766, 'gamma1': 0.0058839461349999775, 'max_depth': 24.0, 'max_depth0': 3.56785768135666, 'max_depth1': 12.430484680638354, 'min_samples_leaf': 42.0, 'min_samples_leaf0': 34.68476584243108, 'min_samples_split': 9.0, 'min_samples_split0': 49.815167439368665, 'n_estimators': 152.12980722678245, 'n_estimators1': 93.94139480702563}
# -0.5543478260869565
# Stacking Model Accuracy: 0.51
# Kappa系数为： 0.5899209049533825
# 宏查准率为: 0.6164053176553176
# 宏查全率为: 0.4469318892269883
# f1指数为： 0.4895071296905314

# {'C': 27.543547297496424, 'eta': 0.21717986628081543, 'gamma': 8.976201976219766, 'gamma1': 0.11979837614346742, 'max_depth': 16.0, 'max_depth0': 2.6075774982829754, 'max_depth1': 15.98883500038394, 'min_samples_leaf': 34.0, 'min_samples_leaf0': 29.740186108531162, 'min_samples_split': 13.0, 'min_samples_split0': 25.473614927569606, 'n_estimators': 183.2110706143528, 'n_estimators1': 143.77748552263265}
# -0.5579710144927537
# Stacking Model Accuracy: 0.54
# Kappa系数为： 0.616974876917824
# 宏查准率为: 0.6371583659325942
# 宏查全率为: 0.4817627887859688
# f1指数为： 0.528627946612435

# {'C': 27.14107769642308, 'eta': 0.26364634795583, 'gamma': 16.036007620175056, 'gamma1': 0.050328537772262164, 'max_depth': 15.0, 'max_depth0': 6.5170184406008325, 'max_depth1': 14.893617902163388, 'min_samples_leaf': 27.0, 'min_samples_leaf0': 18.63526359527014, 'min_samples_split': 91.0, 'min_samples_split0': 21.65125687333606, 'n_estimators': 171.92140148330225, 'n_estimators1': 137.2759568836941}
# -0.5471014492753623
# Stacking Model Accuracy: 0.54
# Kappa系数为： 0.6073226849844944
# 宏查准率为: 0.6556251579136819
# 宏查全率为: 0.46398679143879235
# f1指数为： 0.51202647695185

# {'C': 11.06376776219203, 'eta': 0.2340406628933872, 'gamma': 8.099864541731261, 'gamma1': 0.09666261463283254, 'max_depth': 11.0, 'max_depth0': 6.6741998562763465, 'max_depth1': 15.943992149276168, 'min_samples_leaf': 35.0, 'min_samples_leaf0': 15.80294784664482, 'min_samples_split': 109.0, 'min_samples_split0': 24.20106202253844, 'n_estimators': 59.579276924828605, 'n_estimators1': 92.64952067589567}
# -0.5471014492753623
# Stacking Model Accuracy: 0.53
# Kappa系数为： 0.5602169666146442
# 宏查准率为: 0.5818987170321145
# 宏查全率为: 0.4572487045913491
# f1指数为： 0.49500907108215997

#增加softmax参数
# {'C': 14.203670450055746, 'C2': 61.91971683932062, 'eta': 0.21682297120924746, 'gamma': 5.021447325561955, 'gamma1': 0.2857194221274355, 'max_depth': 2.0, 'max_depth0': 6.449081499318725, 'max_depth1': 5.244030619312969, 'min_samples_leaf': 3.0, 'min_samples_leaf0': 12.866515027503347, 'min_samples_split': 45.0, 'min_samples_split0': 33.711404285256876, 'n_estimators': 75.47588707402599, 'n_estimators1': 188.87755010782337}
# {'C': 26.787418930625257, 'C2': 38.859737207132994, 'eta': 0.13899272319783965, 'gamma': 7.329212772351073, 'gamma1': 0.0031683121869153785, 'max_depth': 6.0, 'max_depth0': 2.9385048530425872, 'max_depth1': 10.806055135290613, 'min_samples_leaf': 25.0, 'min_samples_leaf0': 45.63455860837054, 'min_samples_split': 14.0, 'min_samples_split0': 30.722719117471776, 'n_estimators': 229.95552766636115, 'n_estimators1': 150.86853714441028}
# -0.5471014492753623
# Stacking Model Accuracy: 0.55
# Kappa系数为： 0.617732558139535
# 宏查准率为: 0.6352289453793213
# 宏查全率为: 0.5179140360955203
# f1指数为： 0.5587828550279123

# {'C': 49.6127002491916, 'C2': 47.149187658822804, 'eta': 0.3285479907152741, 'gamma': 2.3608383247145275, 'gamma1': 0.23706722328211038, 'max_depth': 15.0, 'max_depth0': 3.8588426019046493, 'max_depth1': 17.61258900006751, 'min_samples_leaf': 73.0, 'min_samples_leaf0': 17.982585877511994, 'min_samples_split': 67.0, 'min_samples_split0': 25.260923691710584, 'n_estimators': 9.026615045984062, 'n_estimators1': 140.51666580655368}
# -0.5507246376811594
# tacking Model Accuracy: 0.52
# Kappa系数为： 0.5425459902843035
# 宏查准率为: 0.5182703557595274
# 宏查全率为: 0.49958176082966266
# f1指数为： 0.5062747243548348

# {'C': 6.813299013470871, 'C2': 13.514665697758254, 'eta': 0.1888103946069394, 'gamma': 16.2681237819848, 'gamma1': 0.2848845472765471, 'max_depth': 16.0, 'max_depth0': 3.578643130171182, 'max_depth1': 14.213530399837943, 'min_samples_leaf': 103.0, 'min_samples_leaf0': 3.53131147622892, 'min_samples_split': 3.0, 'min_samples_split0': 33.71146397703262, 'n_estimators': 89.67618042870647, 'n_estimators1': 90.41042850504758}
# -0.5507246376811594
# Stacking Model Accuracy: 0.55
# Kappa系数为： 0.6081537951607556
# 宏查准率为: 0.587313251161972
# 宏查全率为: 0.4672089055061368
# f1指数为： 0.491571212009318

# {'C': 32.362280940805135, 'C2': 13.57615674800392, 'eta': 0.2554965738157603, 'gamma': 12.606035813447381, 'gamma1': 0.019694596054124602, 'max_depth': 3.0, 'max_depth0': 5.332176355571758, 'max_depth1': 14.316345772766976, 'min_samples_leaf': 46.0, 'min_samples_leaf0': 26.47699462905156, 'min_samples_split': 51.0, 'min_samples_split0': 13.724767616320783, 'n_estimators': 7.280673561971862, 'n_estimators1': 198.86186395708825}
# -0.5507246376811594
# Stacking Model Accuracy: 0.53
# Kappa系数为： 0.5671501346571919
# 宏查准率为: 0.6176458400942592
# 宏查全率为: 0.4926628973071549
# f1指数为： 0.5331136071501182

# {'C': 19.527363707352784, 'C2': 28.86789707938741, 'eta': 0.47955927173855306, 'gamma': 14.635896344460855, 'gamma1': 0.044068829322820025, 'max_depth': 19.0, 'max_depth0': 7.670697326227156, 'max_depth1': 15.031545572414258, 'min_samples_leaf': 72.0, 'min_samples_leaf0': 30.425722506977916, 'min_samples_split': 44.0, 'min_samples_split0': 33.6813676515981, 'n_estimators': 208.45549749084304, 'n_estimators1': 213.18771500423185}
# -0.5579710144927537
# ng Model Accuracy: 0.54
# Kappa系数为： 0.5709737245526956
# 宏查准率为: 0.5836875693294811
# 宏查全率为: 0.48801526273682966
# f1指数为： 0.5165212046591993