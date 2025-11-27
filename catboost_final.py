# %% [markdown]
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import pandas as pd
#分块处理大文件
# chunk_iter = pd.read_csv("data/train_data.csv", chunksize=10000)
# chunks = [chunk for chunk in chunk_iter]
# train_data = pd.concat(chunks)
train_data = pd.read_csv("train/train.csv")
test_data = pd.read_csv("testaa/testaa.csv")

train_data.info()

# %%
###   特征工程
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 删除zip-code
train_data.drop('zip_code',axis=1, inplace=True)
test_data.drop('zip_code',axis=1, inplace=True)

# 1. 初始自动分离（基于数据类型）
numerical_fea = train_data.select_dtypes(exclude=['object']).columns.tolist()
category_fea = train_data.select_dtypes(include=['object']).columns.tolist()
numerical_fea.remove('label')  # 移除标签列

# 2. 手动调整特定字段类型
features_to_convert = ['title', 'career','residence']  # 需重分类的特征

# 将指定特征从类别型移至数值型
for feature in features_to_convert:
    if feature in category_fea:
        category_fea.remove(feature)
        numerical_fea.append(feature)

# 3. 缺失值处理
# 数值特征：中位数填充 balance_limit
train_data[numerical_fea] = train_data[numerical_fea].fillna(train_data[numerical_fea].median())
test_data[numerical_fea] = test_data[numerical_fea].fillna(test_data[numerical_fea].median())
# 类别特征：众数填充/-1填充 career
train_data[category_fea] = train_data[category_fea].fillna(-1)
test_data[category_fea] = test_data[category_fea].fillna(-1)


# 4. 类别特征编码
def level_to_num(level):
    grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    grade = level[0]              # A/B/C/D/E
    sub = int(level[1])           # 0-5
    return grade_map[grade] * 6 + sub
train_data['level_num'] = train_data['level'].apply(level_to_num)
test_data['level_num'] = test_data['level'].apply(level_to_num)
train_data.drop('level', axis=1, inplace=True)
test_data.drop('level', axis=1, inplace=True)
numerical_fea.append('level_num')
category_fea.remove('level')


# %%
# 对时间进行处理
import datetime
current_time = datetime.datetime.now()  # 或定义一个基准时间
original_time = ['issue_time', 'record_time', 'history_time']
for time_col in original_time:
    train_data[time_col + '_dt'] = pd.to_datetime(train_data[time_col], unit='s')  # Unix秒时间戳
    test_data[time_col + '_dt'] = pd.to_datetime(test_data[time_col], unit='s')
    train_data[time_col + '_age_days'] = (current_time - train_data[time_col + '_dt']).dt.days
    test_data[time_col + '_age_days'] = (current_time - test_data[time_col + '_dt']).dt.days
    numerical_fea.remove(time_col)
    numerical_fea.append(time_col + '_age_days')
    train_data.drop(time_col + '_dt',axis=1, inplace=True)
    test_data.drop(time_col + '_dt',axis=1, inplace=True)


train_data.drop(original_time,axis=1, inplace=True)
test_data.drop(original_time,axis=1, inplace=True)

#cat feature处理
catfeature_list = [col for col in train_data.columns if col != "label"]


# %%
def addStatementFeature(df, filepath):
    df_statement = pd.read_csv(filepath)
    
    # 方法1：保留 'id' 并排除 'label'
    cols_to_merge = [col for col in df_statement.columns if col != 'label']  # 保留 'id'
    
    # 方法2：显式添加 'id'（更安全）
    # cols_to_merge = ['id'] + [col for col in df_statement.columns if col not in ['id', 'label']]
    
    # 合并时确保右侧包含 'id'
    merged_df = pd.merge(
        df,
        df_statement[cols_to_merge],  # 此时包含 'id'
        on='id',
        how='left',
        suffixes=('', '_statement')
    )
    return merged_df

train_data = addStatementFeature(train_data,'train/train_statement_feature.csv')
test_data = addStatementFeature(test_data,'testaa/testaa_statement_feature.csv')
'''
catfeature_list.extend(['income_count', 'expense_count', 'big_income_count',
     'negative_balance_count'])
'''
for fea in catfeature_list:
    train_data[fea] = train_data[fea].astype('int64')
    test_data[fea] = test_data[fea].astype('int64')
 

# %%
category_fea

# %%
numerical_fea

# %%
catfeature_list

# %%
train_data.head()

# %%
from sklearn.model_selection import train_test_split
# 1. 删除id列（正确用法）
train_data.drop('id', axis=1, inplace=True)  # 不赋值，直接修改原对象
catfeature_list.remove('id')
feature_list = [col for col in train_data.columns if col != "label"]
X_train, X_validation, y_train, y_validation = train_test_split(train_data.loc[:, feature_list], train_data.loc[:, 'label'], test_size=0.2 , random_state=2000)

# %%
X_train.head()

# %%
# 3. 模型建立与训练
# 确保 Notebook 内联显示图形
%matplotlib inline 
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=2000,
                           #task_type='GPU',
                           #bootstrap_type='Poisson',
                           task_type='CPU',
                           cat_features=catfeature_list,
                           eval_metric='AUC',
                           logging_level='Verbose',
                           learning_rate=0.03,
                           depth=6, 
                           l2_leaf_reg=5,
                           loss_function='Logloss',
                            early_stopping_rounds=300,
                           scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train == 1])),
                            random_seed= 42
                           )
model.fit(X_train.loc[:, feature_list], y_train, 
          eval_set=(X_validation.loc[:, feature_list], y_validation), plot=True)

plt.show()


# %%
# 4.测试集预测
preds = model.predict_proba(test_data[feature_list])
preds


# %%

test_data['label'] = preds[:, 1]  # 取正类的概率存储
test_data[['id', 'label']].to_csv('submission.csv', index=False)



