import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek

start = time.time()  # Python3.8不支持clock了，使用timer.perf_counter()
# 导入数据，分离特征与输出

file = r'C:\Users\吴振波\Desktop\实验数据集_OpenMLselect3.csv'
cred = pd.read_csv(file)
# df_encoding = df.encode('utf-8')
cred_df = pd.DataFrame(cred)

pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})
# cred.replace({'Sector STRING':{'Transportation':0,'Technology':1,'PublicUtilities':2,'Miscellaneous':3,'Health Care':4,'Finance':5,'Energy':6,'Consumer Services':7,'Consumer Non-Durables':8,'Consumer Durables':9,'Capital Goods':10,'Basic Industries':11}})
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']



#
# 划分训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
## 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

#线性核
linear_svm = SVC(kernel='poly', C=7.8, random_state=0)
linear_svm.fit(X_resampled, y_resampled)
#
# 基于RBF非线性SVM模型训练
rbf_svm = SVC(kernel='rbf', C=33.0480, gamma=0.1539, random_state=0,probability=True)
rbf_svm.fit(X_resampled, y_resampled)

# 模型评估
linear_y_pred = linear_svm.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_y_pred)
#
rbf_y_pred = rbf_svm.predict(X_test)

#获取预测概率
y_pred_proba = rbf_svm.predict_proba(X_test)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)

rbf_accuracy = accuracy_score(y_test, rbf_y_pred)
#

print()
print('线性SVM模型模型评估：')
print('参数C:', linear_svm.C)
print('分类准确度accuracy: ', linear_accuracy)
print()
print('RBF非线性SVM模型评估：')
print('参数C:{:f},gamma:{:.4f}'.format(rbf_svm.C, rbf_svm.gamma))
print('分类准确度accuracy: ', rbf_accuracy)

#{'C': 33.04802550082227, 'gamma': 0.15396618777192939}
# 分类准确度accuracy:  0.38095238095238093
# 0.14687499999999998
# 0.28425925925925927
# 0.2660135841170324
# 0.26365335115335115
# 程序运行时间为: 0.031142473220825195 Seconds


# print(y_test)
# print(rbf_y_pred)
# #模型评估，计算海明距离，大小越接近零说明模型泛化能力越强；越接近1反之
# from sklearn.metrics import hamming_loss
# ham_distance = hamming_loss(y_test,rbf_y_pred)
# print(ham_distance)


from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test,rbf_y_pred) #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print(kappa)
#
from sklearn.metrics import precision_score,recall_score,f1_score
#计算宏查准率
macro_precision = precision_score(y_test, rbf_y_pred, zero_division=1,average="macro")
print(macro_precision)
#计算宏查全率
macro_recall = recall_score(y_test, rbf_y_pred, zero_division=1,average="macro")
print(macro_recall)
#计算f1指数
macro_f1 = f1_score(y_test, rbf_y_pred, zero_division=1,average="macro")
print(macro_f1)

#就算宏平均AUC
# from sklearn import svm
# import numpy as np
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsOneClassifier
# # 使用OneVsRestClassifier处理多分类问题
# clf = svm.SVC(probability=False, decision_function_shape='ovo')
# #训练模型
# ovr_clf = OneVsOneClassifier(clf)
# ovr_clf.fit(X_train, y_train)
# # 获取每个分类器的decision_function结果
# y_score = np.asarray([clf.decision_function(X_test) for clf in ovr_clf.estimators_])
# # 对于OvR策略，我们需要取每行的最大值作为该样本属于对应类别的得分
# y_score = y_score.max(axis=0)
# # 将标签二值化
# y_test_binarize = label_binarize(y_test, classes=[1,2,3,4,5,6,7,8,9])
# # 计算每个类别的AUC
#
# aucs = []
# for i in range(y_test_binarize.shape[1]):
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_binarize[:, i], y_score)
#     auc = metrics.auc(fpr, tpr)
#     aucs.append(auc)
# # 计算宏AUC（Macro-AUC）
# macro_auc = np.mean(aucs)
# print(f'Macro-AUC: {macro_auc:.3f}')
# macro_auc = roc_auc_score(y_test, rbf_y_pred,average='macro')
# print('Macro-average AUC: %d' % macro_auc)


end = time.time()
print('程序运行时间为: %s Seconds' % (end - start))
# print('真实分类：', list(y_test))
# print('预测分类', list(linear_y_pred))
#
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# # 绘制混淆矩阵
"""该方法将在1.2版本之后移除，建议使用下面的展示方法
plot_confusion_matrix(model, X_test, y_test)
plt.show()
"""
# cm = confusion_matrix(y_test, rbf_y_pred, labels=model.classes_)
# disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
# disp.plot()
# plt.show()
# # 混淆矩阵打印输出
# print(cm)

# 参数C:33.048000,gamma:0.1539
# 分类准确度accuracy:  0.39285714285714285
# 0.2553450373718059
# 0.38521986054880797
# 0.4285592998955068
# 0.3657583903551645
# 程序运行时间为: 0.20410585403442383 Seconds

# {'C': 49.74033587948649, 'gamma': 0.24755184500503263}
# 分类准确度accuracy:  0.38095238095238093
# 0.22690265486725658
# 0.36002666923719556
# 0.3420715778474399
# 0.33593641671227875

# {'C': 20.95666480510945, 'gamma': 0.199973668058762}
# 分类准确度accuracy:  0.38095238095238093
# 0.23942190492773818
# 0.3852247168036642
# 0.3868926332288401
# 0.35256047642617516

# {'C': 49.86369711147436, 'gamma': 0.1536688339053653}
# 分类准确度accuracy:  0.39285714285714285
# 0.2540484067560509
# 0.37519322782480674
# 0.39263975966562176
# 0.3530508772648732

# {'C': 2.3941255942478348, 'gamma': 0.39777648153868234}
# 分类准确度accuracy:  0.38095238095238093
# 0.2498712004121587
# 0.4238153594771242
# 0.4399229362591432
# 0.3823556475730389
