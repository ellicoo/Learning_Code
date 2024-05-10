'''
1. 开发一个功能菜单，提示用户可以实现的功能
2. 在功能菜单的基础上添加一个用户输入功能
3. 系统可以根据用户的输入实现不同的功能
【1】添加学生
【2】删除学生
【3】修改学生
【4】查询学生
【5】遍历所有学生信息
【6】持久化保存学生信息
【7】退出系统
在Python代码中，为了让代码可以重复使用，我们可以考虑把程序封装函数中 => （① 模块化编程 ② 代码重用）
'''
# 1. 封装一个menu函数，用于打印系统功能菜单
def menu():
    print('-' * 40)
    print('------ 欢迎使用传智教育学生管理系统V1.0 ------')
    print('【1】添加学生信息')
    print('【2】删除学生信息')
    print('【3】修改学生信息')
    print('【4】查询学生信息')
    print('【5】遍历所有学生信息')
    print('【6】持久化保存学生信息')
    print('【7】退出系统')
    print('-' * 40)

# 4. 定义一个全局变量 => students => []列表，专门用于保存所有学生信息
students = []

# 5. 定义一个add_student()函数，用于实现向系统中添加学员信息
def add_student():
    name = input('请输入您要添加学生的姓名：')
    age = int(input('请输入您要添加学生的年龄：'))
    mobile = input('请输入您要添加学生的电话：')
    # 定义一个空字典 => 用于保存每个学生的信息
    student = {}
    student['name'] = name
    student['age'] = age
    student['mobile'] = mobile
    # student = {'name':'Tom', 'age':23, 'mobile':'10086'}
    # 声明全局变量
    global students
    students.append(student)  # [{}, {}, {}]
    print(students)  # 测试代码
    print('恭喜您，信息已添加成功！')

# 6. 定义一个del_student()函数，用于删除指定同学的信息
def del_student():
    # 提示用户输入要删除同学的姓名
    name = input('请输入要删除的同学姓名：')
    # 声明全局变量
    global students
    for i in students:  # [{}, {}, {}]
        # 判断要删除学生名称是否与字典中的name这个key的值相等
        if i['name'] == name:
            students.remove(i)
            print(students)
            print('恭喜您，信息删除成功！')
            break
    else:
        print('很抱歉，您要删除的学生未找到！')

# 7. 封装一个函数edit_student()，用于修改学生信息
def edit_student():
    # 提示用户输入要修改的学生姓名
    name = input('请输入要修改的学生姓名：')
    # 需要声明全局变量
    global students
    # 对students进行遍历，判断是否有要修改的学生
    for i in students:  # [{}, {}, {}]
        if i['name'] == name:
            # 修改其信息 => age、mobile
            i['age'] = int(input('请输入修改后学生的年龄：'))
            i['mobile'] = input('请输入修改后学生的电话：')
            # 打印测试输出
            print(students)
            # 弹出修改成功提示
            print('恭喜您，信息修改成功')
            break
    else:
        print('很抱歉，您要修改的学生并不存在！')

# 8. 定义一个get_student()函数，用于查找指定的同学信息
def get_student():
    # 提示用户输入要查询学生姓名
    name = input('请输入要查询学生的姓名：')
    # 声明全局变量
    global students
    # 使用for循环对其进行遍历
    for i in students:  # [{}, {}, {}]
        # 如果某个同学名称刚好与name相等，则弹出这个字典信息（就是这个同学）
        if i['name'] == name:
            print(i)
            break
    else:
        print('很抱歉，您要查找的学生信息不存在！')

while True:
    menu()
    # 2. 想个办法，阻塞死循环代码，让其停下来 => input()
    user_num = int(input('请输入【】您要执行的功能编号：'))
    # 3. 根据用户的输入，执行不同的功能 => if...elif...else
    if user_num == 1:
        add_student()
    elif user_num == 2:
        del_student()
    elif user_num == 3:
        edit_student()
    elif user_num == 4:
        get_student()
    elif user_num == 5:
        print('遍历所有学生信息')
    elif user_num == 6:
        print('持久化保存学生信息')
    elif user_num == 7:
        print('感谢您使用传智教育学生管理系统V1.0，欢迎下次光临！')
        break
    else:
        print('很抱歉，您输入的功能编号有误，请重新输入！')