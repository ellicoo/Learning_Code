'''
定义学生管理系统类，主要用于实现对学生对象的增删改查操作。
类中有个属性self.students = []，列表主要用于大批量数据的存储。
self.students = [s1, s2, s3]，这里s1，s2，s3等等都是由Student学生类产生的学生对象
'''
# 1、把Student.py中的Student类导入当前页面
from Student import Student
# 2、定义一个学生管理系统类
class StudentCMS(object):
    # 3、定义一个students属性，列表类型，将来用于保存所有学生的信息
    def __init__(self):
        self.students = []
    # 4、定义一个menu()方法，用于打印开始菜单
    @staticmethod
    def menu():
        print('-' * 60)
        print('-----------------欢迎使用传智教育学生管理系统V2.0-----------------')
        print('-----------------【1】添加学生信息')
        print('-----------------【2】删除学生信息')
        print('-----------------【3】修改学生信息')
        print('-----------------【4】查询学生信息')
        print('-----------------【5】遍历所有学生信息')
        print('-----------------【6】保存数据信息到文件')
        print('-----------------【7】退出学生管理系统')
        print('-----------------欢迎使用传智教育学生管理系统V2.0-----------------')
        print('-' * 60)

    # 10、定义一个add_student()方法，实现添加学生信息
    def add_student(self):
        # 第一步：接收外部的输入信息（姓名、年龄、电话号码）
        name = input('请输入学生的姓名：')
        age = int(input('请输入学生的年龄：'))
        mobile = input('请输入学生的电话：')
        # 第二步：创建一个学生对象
        # 第三步：把创建好的学生对象写入到自身的self.students列表中 => [s1, s2, s3]
        self.students.append(Student(name, age, mobile))  # [s1, s2, s3]
        # 第四步：提示用户学生信息添加成功
        print(self.students)
        print('恭喜您，信息录入成功！')

    # 11、定义一个del_student()方法，实现删除学生信息
    def del_student(self):
        # 第一步：接收要删除学生的名称
        name = input('请输入您删除学生的姓名：')
        # 第二步：对所有学生进行遍历，查找与之匹配的学生
        for i in self.students:
            # 第三步：如果匹配到，则执行删除操作
            if i.name == name:
                self.students.remove(i)
                print('恭喜您，信息删除成功！')
                break
        # 第四步：如果未匹配到，则循环正常结束
        else:
            print('很抱歉，您删除的学生不存在！')

    # 12、定义一个mod_student()函数，用于修改学生信息
    def mod_student(self):
        # 第一步：接收要修改学生的姓名
        name = input('请输入要修改学生的姓名：')
        # 第二步：对所有学生进行遍历，查找与之匹配的学生
        for i in self.students:
            # 第三步：如果匹配到，则执行修改操作
            if i.name == name:
                i.age = int(input('请输入要修改学生的年龄：'))
                i.mobile = input('请输入要修改学生的电话：')
                print('恭喜您，信息修改成功！')
                break
        # 第四步：如果未匹配到，则循环正常结束
        else:
            print('很抱歉，您要修改的学生信息不存在！')

    # 13、定义一个show_student()函数，用于打印学生信息
    def show_student(self):
        # 第一步：接收要查询学生的信息
        name = input('请输入要查询学生姓名：')
        # 第二步：对整个列表进行遍历，查找是否有与之匹配的学生
        for i in self.students:
            # 第三步：如果匹配到，则打印对象信息并终止整个循环
            if i.name == name:
                print(i)  # 返回的是内存地址？答：不是，因为我们在Student类中定义一个__str__()
                break
        # 第四步：如果未匹配到，则执行else结构
        else:
            print('很抱歉，您要查找的学生不存在！')

    # 14、定义一个show_all_students()函数，用于遍历所有学生信息
    def show_all_students(self):
        for i in self.students:
            print(i)

    # 15、定义一个save_data_to_file()函数，用于保存数据到文件
    def save_data_to_file(self):
        # 第一步：打开文件
        f = open('students.txt', 'w', encoding='utf-8')
        # 第二步：读写文件
        new_list = [i.__dict__ for i in self.students]
        f.write(str(new_list))
        # 第三步：关闭文件
        f.close()
        print('恭喜您，信息已写入文件成功！')

    # 17、定义一个load_data_from_file()函数，用于加载文件中的数据
    def load_data_from_file(self):
        try:
            f = open('students.txt', 'r', encoding='utf-8')
        except:
            f = open('students.txt', 'w', encoding='utf-8')
        else:
            content = f.read()
            if not content:  # 文件中没有数据，空数据，则if条件则为真
                content = '[]'
            # 文件内容不为空，读取文件，去掉两边引号
            new_list = eval(content)
            # 把字典在转换为对象类型，然后把整个列表赋值给变量self.students
            self.students = [Student(i['name'], i['age'], i['mobile']) for i in new_list]
        finally:
            # 无论文件是否存在，直接关闭文件
            f.close()

    # 5、定义一个start()方法，用于启动学生管理系统
    def start(self):
        # 16、加载文件中的数据到self.students
        self.load_data_from_file()
        # 6、定义一个wihle True死循环，让代码可以一直执行
        while True:
            # 7、打印功能菜单
            self.menu()
            # 8、添加一个input()语句
            user_num = int(input('请输入您要执行的功能菜单编号：'))
            # 9、添加一个判断语句
            if user_num == 1:
                self.add_student()  # 添加学生信息
            elif user_num == 2:
                self.del_student()  # 删除学生信息
            elif user_num == 3:
                self.mod_student()  # 修改学生信息
            elif user_num == 4:
                self.show_student()  # 查询学生信息
            elif user_num == 5:
                self.show_all_students()  # 遍历所有学生信息
            elif user_num == 6:
                self.save_data_to_file()  # 保存数据到文件
            elif user_num == 7:
                print('退出系统成功，感谢您使用传智教育学生管理系统V2.0!')
                break