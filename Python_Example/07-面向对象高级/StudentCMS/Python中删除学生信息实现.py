from Student import Student  # 把Student.py文件中的Student类导入当前页面

# 定义一个列表
students = []
s = Student('Tom', 23, '10086')
students.append(s)

s = Student('Rose', 24, '10010')
students.append(s)

print(students)
# -------------------------------------------
# 删除操作实现 => 实现步骤
# 1、定义一个变量，用于接收要删除学员的名称
name = input('请输入要删除学生的姓名：')
# 2、对现有学生列表进行遍历 => 查询哪个对象的name属性刚好与name变量相等
for i in students:
    if i.name == name:
        # 找到了这个对象，删除
        students.remove(i)
        # 删除成功后，强制终止循环执行
        print('恭喜您，信息删除成功')
        print(students)
        break
else:
    print('很抱歉，您要删除的学生不存在！')