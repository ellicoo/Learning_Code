'''
在Python中，集合一共有3种操作方法
增加操作：集合.add(元素)
删除操作：集合.remove(元素)
查询操作：元素 in 集合，如果元素存在集合中，返回True；反之就返回False

注意：add()、remove()本身只返回None，所有的操作都是针对原集合的，所以操作完成后，要直接打印集合变量

疑问：为什么集合没有修改方法？
答：集合中的元素没有顺序所以就导致集合无法定位到具体的元素，所以就没有修改方法了
'''
# 1、定义一个空集合
set1 = set()
# 2、向其内部追加元素 => 10, 20, 30
set1.add(10)
set1.add(20)
set1.add(30)
print(set1)
# 3、删除20这个元素
set1.remove(20)
print(set1)
# 4、判断10这个元素是否出现在集合中
if 10 in set1:
    print('exists')
else:
    print('not exists')