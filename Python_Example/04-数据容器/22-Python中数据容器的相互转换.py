'''
回顾基本类型转换，如int、float、str
int()：把其他类型转换int类型
float()：把其他类型转换为float类型
str()：把其他类型转换为str类型
eval()：把字符串两边引号去掉，转换为原数据类型
--------------------------------------
list()：把其他容器类型转换为列表类型
tuple()：把其他容器类型转换为元组类型
set()：把其他容器类型转换为set集合类型
'''
# 1、定义一个元组，将其转换为list列表类型
tuple1 = (10, 20, 30)
list1 = list(tuple1)
print(list1)
print(type(list1))

# 2、把列表转换为元组类型
list2 = [10, 100, 1000]
tuple2 = tuple(list2)
print(tuple2)
print(type(tuple2))

# 3、定义一个列表，将其转换为set集合类型 => 去重
list3 = [10, 20, 30, 20, 30]
set3 = set(list3)
print(set3)
print(type(set3))