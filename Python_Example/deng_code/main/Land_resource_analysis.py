import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import entropy

# 读取CSV数据
data = pd.read_csv('/Users/otis/PycharmProjects/Python_Example/deng_code/data/data_detail.csv')

# 提取感兴趣的指标数据
interested_columns = ['地区', '时间', '建设用地面积增速（%）', '地均产值(万元/KM2）', '地均财政支出(万元/KM2）', '地均固定资产投资(万元/KM2）', '土地总产值']
interested_data = data[interested_columns]

# 根据地区和时间分组，计算平均值
grouped_data = interested_data.groupby(['地区', '时间']).mean().reset_index()

# 数据标准化
numeric_data = grouped_data.drop(columns=['地区', '时间', '土地总产值'])  # 只保留数值列
normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

# 计算熵值
entropy_values = normalized_data.apply(entropy, axis=0)

# 计算权重
weights = 1 - entropy_values / np.log2(len(numeric_data.columns))

# 将权重添加到数据中
for i, col in enumerate(numeric_data.columns):
    grouped_data[col + '权重'] = weights[i]

# 计算综合评价结果
weighted_data = normalized_data * weights.values
grouped_data['综合评价结果'] = weighted_data.sum(axis=1)

# 输出结果
display(data)

