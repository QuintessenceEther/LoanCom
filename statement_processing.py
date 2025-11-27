# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
def load_time_process(filepath):
    # 加载数据
    df = pd.read_csv(filepath)
    # 转换时间戳为datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # 创建交易日期列，此处3.10必须normalize
    df['date'] = df['time'].dt.normalize()
    
    # 添加时间特征
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5  # 5和6是周六周日
    return df
def statement_feature_engineering(df):
    # 按客户分组
    grouped = df.groupby('id')    
    # 基础特征
    features = pd.DataFrame(index=df['id'].unique())
    features.index.name = 'id'
    # 1. 基本收支统计
    features['total_income'] = grouped.apply(lambda x: x[x['direction'] == 1]['amount'].sum())
    features['total_expense'] = grouped.apply(lambda x: x[x['direction'] == 0]['amount'].sum())
    features['net_income'] = features['total_income'] - features['total_expense']
    features['income_count'] = grouped.apply(lambda x: (x['direction'] == 1).sum())
    features['expense_count'] = grouped.apply(lambda x: (x['direction'] == 0).sum())
    features['total_tx_count'] = features['income_count'] + features['expense_count']       
    # 2. 时间维度特征
    min_date = grouped['date'].min()
    max_date = grouped['date'].max()
    features['active_days'] = (max_date - min_date).dt.days + 1
    features['tx_frequency'] = features['total_tx_count'] / features['active_days']
    now_date = pd.Timestamp.now().normalize()
    features['last_tx_days_ago'] = (now_date - max_date).dt.days
    # 4. 收支平衡特征
    features['income_expense_ratio'] = features['total_income'] / (features['total_expense'] + 1e-6)
    features['avg_income'] = features['total_income'] / features['income_count']
    features['income_std'] = grouped.apply(lambda x: x[x['direction'] == 1]['amount'].std())
    features['expense_std'] = grouped.apply(lambda x: x[x['direction'] == 0]['amount'].std())
    # 5. 大额交易特征
    features['big_income_count'] = grouped.apply(
        lambda x: ((x['direction'] == 1) & (x['amount'] > x['amount'].mean() * 2)).sum()
    )
    features['big_income_ratio'] = features['big_income_count'] / features['income_count']
       # 6. 余额模拟特征
    def calculate_balance(group):
        group = group.sort_values('time')
        balance = 0
        min_balance = float('inf')
        negative_count = 0
        
        for _, row in group.iterrows():
            if row['direction'] == 1:  # 收入
                balance += row['amount']
            else:  # 支出
                balance -= row['amount']
                
            if balance < min_balance:
                min_balance = balance
            if balance < 0:
                negative_count += 1
                
        return pd.Series({
            'min_balance': min_balance,
            'final_balance': balance,
            'negative_balance_count': negative_count,
            'negative_balance_ratio': negative_count / len(group)
        })
    
    balance_features = grouped.apply(calculate_balance)
    features = features.join(balance_features)
    # 填充空值
    features.fillna(0, inplace=True)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    #索引保留
    features = features.reset_index().rename(columns={'index': 'id'})
    return features

# %%
# 对流水文件进行处理
df_train = load_time_process('train/train_bank_statement.csv')
df_test = load_time_process('testaa/testaa_bank_statement.csv')
df_train = statement_feature_engineering(df_train)
df_test = statement_feature_engineering(df_test)
train_data = pd.read_csv("train/train.csv")
test_data = pd.read_csv("testaa/testaa.csv")
merged_df_train = pd.merge(
    df_train,                  # 主表
    train_data[['id', 'label']], # 仅选取标签表的 id 和 label 列
    on='id',                   # 根据 id 列合并
    how='left'                 # 保留 df_train 所有行，未匹配的 label 填充为 NaN
)

merged_df_train.to_csv('train/train_statement_feature.csv',index=False)
df_test.to_csv('testaa/testaa_statement_feature.csv',index=False)


