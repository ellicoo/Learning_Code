'''
定义Student学生类，包含name、age、mobile属性信息
'''
class Student(object):
    # 定义__init__()魔术方法
    def __init__(self, name, age, mobile):
        self.name = name
        self.age = age
        self.mobile = mobile

    # 打印输出对象 => print(s1)返回内存地址 => 显示对象自身信息 => __str__()
    def __str__(self):
        return f"姓名：{self.name}，年龄：{self.age}，电话：{self.mobile}"