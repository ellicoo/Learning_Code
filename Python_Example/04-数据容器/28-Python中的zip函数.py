# 语法： zip(*iterables) *iterables 是一个可变参数，表示要打包的多个可迭代对象

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# 使用 zip 函数将两个列表打包成元组的迭代器
zipped = zip(list1, list2)
print(zipped)
print(type(zipped))

# 将迭代器转换为列表，以便查看结果
result = list(zipped)
print(result)

"""
zip() 函数常用于在迭代时同时遍历多个可迭代对象，并且在需要时将它们组合在一起。
这在需要同时处理多个列表或其他可迭代对象的情况下非常有用。例如，在迭代字典时，
zip() 可以同时遍历字典的键和值：
"""

# my_dict = {'a': 1, 'b': 2, 'c': 3}
my_dict = dict(result)
for key, value in zip(my_dict.keys(), my_dict.values()):
    print(key, value)

for key in my_dict.keys():
    print(key)

print("-----------------")
total_feature = ['a', 'b', 'c']
unique_val = [4, 2, 7]
a = dict(zip(unique_val, range(0, len(unique_val) + 0)))
# a=range(0,len(unique_val) + 0)
print(a)
print("----------------")
a = list(zip(unique_val, range(2, len(unique_val) + 2)))
print(a)
print("----------------")
a = dict(zip(unique_val, range(2, len(unique_val) + 2)))
print(a)

print("---------final----------")

feature_dict = {'ps_ind_01': {2: 0, 1: 1, 5: 2, 0: 3, 4: 4, 3: 5, 6: 6, 7: 7},
                'ps_ind_02_cat': {2: 8, 1: 9, 4: 10, 3: 11, -1: 12},
                'ps_ind_03': {5: 13, 7: 14, 9: 15, 2: 16, 0: 17, 4: 18, 3: 19, 1: 20, 11: 21, 6: 22, 8: 23, 10: 24},
                'ps_ind_04_cat': {1: 25, 0: 26, -1: 27},
                'ps_ind_05_cat': {0: 28, 1: 29, 4: 30, 3: 31, 6: 32, 5: 33, -1: 34, 2: 35},
                }

for col in feature_dict.keys():
    a = feature_dict[col].map(feature_dict.values())
    print(a)
