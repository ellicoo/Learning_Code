from Student import Student


students = []
s1 = Student('Tom', 23, '10086')
s2 = Student('Rose', 24, '10010')
students.append(s1)
students.append(s2)

# 把学生对象转换为字典类型 => 多个，必须遍历实现
# newlist = []
# for i in students:
#     newlist.append(i.__dict__)
#
# print(newlist)

# 把以上程序使用列表推导式转换一下
newlist = [i.__dict__ for i in students]
print(newlist)