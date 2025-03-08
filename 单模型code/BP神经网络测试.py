import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import pandas as pd
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
# cred.replace({'企业信用等级':{'C':0,'CC':1,'CCC':2,'B':3,'BB':4,'BBB':5,'A':6,'AA':7,'AAA':8}})

# 2.提取特征变量与目标变量
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']


label_encoder = LabelEncoder()
# 使用fit_transform方法将文本标签转换为整数
integer_y = label_encoder.fit_transform(y)

# 3.划分训练集与测试集
from sklearn.model_selection import train_test_split
# 因为我们有三个类别，所以我们需要将标签转换为one-hot编码
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, integer_y, test_size=0.3, random_state=0)

y_one_hot = to_categorical(integer_y, num_classes=9)
# print(y_one_hot)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=0)



# smote_tomek = SMOTETomek(random_state=0)
# X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 初始化BP神经网络模型
model = Sequential()
model.add(Dense(11, input_dim=13, activation='leaky_relu'))  # 输入层到隐藏层，假设有10个神经元
model.add(Dense(9, activation='softmax'))  # 隐藏层到输出层，9个神经元对应9个类别，使用softmax激活函数

# 编译模型，设置优化器、损失函数和评估指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=95 ,batch_size=7, verbose=1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 使用模型进行预测
y_pred = model.predict(X_test)


# 因为预测结果是概率分布，所以我们需要取概率最大的类别作为预测结果
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_classes, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)
# 输出预测结果（如果需要）
print("Predicted classes:", y_pred_classes)

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test1,y_pred_classes,) #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print("Kappa系数为：",kappa)
#
from sklearn.metrics import precision_score,recall_score,f1_score
#计算宏查准率
macro_precision = precision_score(y_test1, y_pred_classes, zero_division=1,average="macro")
print("宏查准率为:",macro_precision)
#计算宏查全率
macro_recall = recall_score(y_test1, y_pred_classes, zero_division=1,average="macro")
print("宏查全率为:",macro_recall)
#计算f1指数
macro_f1 = f1_score(y_test1, y_pred_classes, zero_division=1,average="macro")
print("f1指数为：",macro_f1)

# {'batch_size': 40.08785242401571, 'epochs': 36.62145032507544, 'n_p': 5.582665377378637}

# {'batch_size': 71.09296075823065, 'epochs': 51.50448575921653, 'n_p': 36.528484091192695}
# Test accuracy: 0.3584
# Kappa系数为： 0.08756294394237485
# 宏查准率为: 0.38376919571656415
# 宏查全率为: 0.2079242031823163
# f1指数为： 0.1735444787770369

# {'batch_size': 43.01198479910549, 'epochs': 11.021969584479217, 'n_p': 30.295990953623054}

#{'batch_size': 50.69466915302625, 'epochs': 104.7635451637678, 'n_p': 45.66888523548419}
# Test accuracy: 0.3923
# Kappa系数为： 0.1553916975597831
# 宏查准率为: 0.2806369612984217
# 宏查全率为: 0.2439393052771117
# f1指数为： 0.23218531562505174

# {'batch_size': 81.68879123931491, 'epochs': 112.19291468826614, 'n_p': 24.96809658229168}
# Test accuracy: 0.3947
# Kappa系数为： 0.14713122201846984
# 宏查准率为: 0.46008924607915525
# 宏查全率为: 0.23793550868725455
# f1指数为： 0.21575271293543005

# {'batch_size': 98.0208757306361, 'epochs': 25.970901438999217, 'n_p': 15.443864732040106}
# Test accuracy: 0.3366
# Kappa系数为： 0.048186994810372585
# 宏查准率为: 0.4087569616272622
# 宏查全率为: 0.3080455303213074
# f1指数为： 0.13344109011199398

#{'batch_size': 7.980703852397423, 'epochs': 95.42255400128607, 'n_p': 11.393405542844864}
# Test accuracy: 0.3995
# Kappa系数为： 0.16622708475790426
# 宏查准率为: 0.47095990931216175
# 宏查全率为: 0.25242140664980595
# f1指数为： 0.2419132867511827
