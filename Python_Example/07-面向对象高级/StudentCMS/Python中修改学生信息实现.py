from Student import Student  # 把Student.py文件中的Student类导入当前页面

# 定义一个列表
students = []
s = Student('Tom', 23, '10086')
students.append(s)

s = Student('Rose', 24, '10010')
students.append(s)

print(students)
# -------------------------------------------------------------------
# 第一步：传入要修改学生的姓名
name = input('请输入您要修改学生的姓名：')
# 第二步：循环列表，判断是否有学生姓名与传入的名称一致
for i in students:
    # 第三步：如果传入名称与循环中的某个名字相同，则代表找到了这个同学
    if i.name == name:
        # 第四步：对其信息进行修改操作 => 对象类型.age，如何修改？
        i.age = int(input('请输入修改后的学生年龄：'))
        i.mobile = input('请输入修改后的学生电话：')
        print('恭喜您，信息修改成功！')
        break
# 第五步：当循环结束，如果还未执行break语句，则代表学生信息不存在
else:
    print('很抱歉，您要修改的学生信息不存在！')