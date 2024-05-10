'''
在Python代码中，内置了很多列表相关方法：
增加 => 列表.append(元素)，可以向列表尾部追加元素。主要针对原列表进行操作，append()方法默认返回None
删除 => 列表.remove(元素)，从原列表中移除指定元素。主要针对原列表进行操作，remove()方法默认返回None
修改 => 列表[索引] = 替换后的值
查询 => 索引下标、for循环遍历、len()、in关键字
len()返回列表中的元素个数
in：元素 in 列表，如果元素的确存在于列表中，返回为True；反之返回为False
'''
# 1、定义一个空列表
list1 = []
# 2、追加孙悟空、猪八戒两个元素
list1.append('孙悟空')
list1.append('猪八戒')
print(list1)
# 3、修改猪八戒为沙悟净
list1[1] = '沙悟净'
print(list1)
# 4、获取列表中的元素个数，判断孙悟空有没有出现在列表中
print(len(list1))
if '孙悟空' in list1:
    print('exists')
else:
    print('not exists')