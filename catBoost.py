import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import pandas as pd
#分块处理大文件
# chunk_iter = pd.read_csv("data/train_data.csv", chunksize=10000)
# chunks = [chunk for chunk in chunk_iter]
# train_data = pd.concat(chunks)
train_data = pd.read_csv("data/processed_train.csv")
test_data = pd.read_csv("data/processed_test.csv")
train_data = train_data.drop(columns='issueDate')
test_data = test_data.drop(columns='issueDate')
#训练测试集划分
from sklearn.model_selection import train_test_split
feature_list = [col for col in train_data.columns if col != "isDefault"]
for fea in feature_list:
    train_data[fea] = train_data[fea].astype('int64')
    test_data[fea] = test_data[fea].astype('int64')
X_train, X_validation, y_train, y_validation = train_test_split(train_data.loc[:, feature_list], train_data.loc[:, 'isDefault'], test_size=0.2 , random_state=2000)
# 3. 模型建立与训练
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=100, cat_features=feature_list, eval_metric='AUC', logging_level='Verbose', learning_rate=0.05, depth=6, l2_leaf_reg=5, loss_function='CrossEntropy')
model.fit(X_train.loc[:, feature_list], y_train, 
          eval_set=(X_validation.loc[:, feature_list], y_validation), plot=True)