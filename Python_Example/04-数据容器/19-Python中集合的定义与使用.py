'''
set集合是一个无序且天生去重的数据序列（数据容器）
基本语法：
① 空集合只能使用set()方法定义
② 有数据的集合可以使用{}方式来进行定义

疑问：{}既可以定义字典也可以定义集合？怎么判别我们定义的到底是字典还是集合类型
答：{}存储的是key:value键值对，则代表定义的就是字典；反之定义的就是集合
'''
# 1、定义一个空集合
set1 = set()
print(type(set1))
# 2、定义一个有数据的集合
set2 = {10, 20, 30, 20, 30}
# 3、打印输出
print(set2)
print(type(set2))
# 4、注意：集合的无序特性
set3 = {'刘备', '关羽', '张飞'}
print(set3)
# 5、由于集合本身是无序的，所以其没有索引下标，要么直接打印，要么使用for循环直接遍历所有元素
for i in set3:
    print(i)