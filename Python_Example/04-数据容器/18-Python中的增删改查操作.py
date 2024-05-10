'''
字典和列表类似，也是一种比较灵活的数据类型，我们可以对字典实现增删改查操作。
① 字典增加/修改操作 => 使用相同的语法
    字典名称[key] = value值
强调：如果字典中不存在这个key，以上操作就是新增元素操作；如果字典中存在这个key，以上就是修改元素操作。
② 删除字典中的指定元素
    del 字典名称[key]
强调：以上操作代表删除字典中指定的key:value键值对，并不是只删除这个key，而是删除这个key:value键值对
③ 查询操作 => 通过3个方法实现，这3个方法分别是
字典.keys()：以类列表（类似列表）方式返回字典中的所有key
字典.values()：以类列表（类似列表）方式返回字典中的所有value
字典.items()：以[(key, value), (key, value), (key, value)]形式返回字典中的key:value键值对
由于以上三个函数都是以类列表方式返回，所以不能直接打印输出，必须和for循环一起使用
'''
# 1、定义一个空字典
student = {}
# 2、追加元素，name:Tom, age:23, mobile:10086
student['name'] = 'Tom'
student['age'] = 23
student['mobile'] = '10086'
# print(student)
# 3、修改元素，把Tom的年龄修改为24岁
student['age'] = 24
print(student)
# 4、删除mobile这个key:value键值对
del student['mobile']
print(student)
# 5、获取字典中的所有key
for key in student.keys():
    print(key)
# 6、获取字典中的所有vlaue
for value in student.values():
    print(value)
# 7、获取字典中的key:value键值对
for key, value in student.items():
    print(key, value)

