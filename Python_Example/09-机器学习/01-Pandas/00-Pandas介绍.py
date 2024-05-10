"""
用途： Pandas是一种强大的数据处理和分析库。它提供了像DataFrame这样的数据结构，用于高效的数据操作并集成了索引功能。
主要特点：
提供了高效的数据操作和分析工具。
数据结构如DataFrame，适用于各种数据操作。
灵活的索引和标签功能。
支持缺失数据的处理。
"""

import pandas as pd

# 创建一个Pandas DataFrame
df = pd.DataFrame({'col1': [[1,2,3], [4,5,6], [7,8,9], 4], 'col2': ['a', 'b', 'c', 'd']})

# 将DataFrame的列转换为列表
col1_list = df['col1'].tolist()
col2_list = df['col2'].tolist()

# 打印转换的列表
print("col1_list:", col1_list)
print("col2_list:", col2_list)


