'''
__str__()：默认情况下，当我们使用print(对象)打印对象时，系统会自动返回对象所指向的内存地址。有了__str__()魔术方法时，系统会自动打印__str__()魔术方法中return的返回结果。

在实际工作中，我们可以用__str__()实现对象信息打印操作
触发条件：当使用print()方法打印对象时，__str__魔术方法会自动触发

# 案例：定一个Person类，为其赋予name和age，在为其定义一个speak()方法，提示自身信息。定义一个__str__魔术方法，打印自身对象相关属性信息

注意：__str__()方法其要求返回的结果必须是一个字符串类型的数据！！！
'''
# 1、定义一个Person类
class Person(object):
    # 2、定义对象属性与方法
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print('i can speak Chinese')

    # 3、定义一个__str__魔术方法
    def __str__(self):
        return f"我的名字为{self.name}，年龄{self.age}岁了！"

# 4、实例化对象，初始化对象属性
p1 = Person('Tom',23)
# 5、打印对象
print(p1)
p1.speak()