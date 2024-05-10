'''
为什么要学习字典？
答：字典可以用于某一事物的存储，比如一个人的信息，一本书的信息，一个商品的信息
基本语法：
字典名称 = {key:value, key:value, key:value}
要求：key在字典中必须是唯一的，如果重复的key，则后面的key会覆盖前面的key
如果key是字符串类型，比如使用单引号或双引号引起来
'''
# 1、定义一个空字典
dict1 = {}  # 空字典（推荐）
dict2 = dict()  # 了解即可
# 2、定义一个有数据的字典 => 姓名：Tom、年龄：23、电话：10086
dict3 = {'name': 'Tom', 'age': 23, 'mobile': '10086'}
dict4 = {1: 10, 2: 20, 3: 30}

print(dict3)
print(dict4)

print(type(dict3))
print(type(dict4))
# 3、字典的访问比较特殊，由于字典中的元素都是基于key:value键值对，所以其没有数字索引，数据的访问都是通过key来实现
print(dict3['name'])
print(dict3['age'])
print(dict3['mobile'])
