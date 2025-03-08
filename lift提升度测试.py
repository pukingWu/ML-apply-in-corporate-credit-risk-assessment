import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 创建一个多分类问题的数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=4,n_clusters_per_class=2, random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

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