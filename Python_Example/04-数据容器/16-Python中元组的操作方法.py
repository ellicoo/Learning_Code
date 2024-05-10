'''
由于元组类型比较特殊，一旦定义完成后，其内部数据就无法修改和删除了。
所有元组的操作都是围绕查询功能

单元素查询 => 元组[索引]
len(元组) ：获取元组中的元素个数
in ：判断元素有没有出现在元组中，出现=>True；没有出现=>False
'''
# 1、定义一个元组
tuple1 = (10, 20, 30, 40)
# 2、打印输出20个这个元素
print(tuple1[1])
# 3、获取整个元组的元素个数
print(len(tuple1))
# 4、判断元素30有没有出现在元组中
if 30 in tuple1:
    print('exists')
else:
    print('not exists')

print('----------------')

# 5、元祖拆包
a = 10
b = 20
(b, a) = (a, b)
print(a)
print(b)
print('-----------------')
# 4、在Python中，元组拆包时，右边的小括号也可以省略
a = 10
b = 20
b, a = a, b
print(a)
print(b)

