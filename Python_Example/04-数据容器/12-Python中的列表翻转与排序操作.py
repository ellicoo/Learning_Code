'''
列表是一个数据容器，所有经常有翻转与排序需求。
翻转（倒序）：列表.reverse()，主要针对原列表进行倒序，reverse()方法本身返回None
排序（升序）：列表.sort()，如果想实现降序排列，可以通过列表.sort(reverse=True)，主要针对原列表进行倒序，sort()方法本身返回None
'''
# 1、定义一个列表，包含元素10, 8, 15, 12
list1 = [10, 8, 15, 12]
# 2、使用至少两种方式实现列表元素翻转
# list1.reverse()
# print(list1)

print(list1[::-1])
# 3、使用sort()方法对列表进行升序与降序排列
list1.sort()
print(list1)

list1.sort(reverse=True)
print(list1)