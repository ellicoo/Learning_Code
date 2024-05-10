'''
__init__()：构造函数，初始化操作
__del__()：析构函数，清理操作
作用：主要是完成清理操作，比如文件操作时，可以用于关闭文件；数据库操作时，可以用于关闭数据库
触发条件：当对象被删除时，__del__()会自动被触发！

案例：定义一个Person类，定义name和age属性以及speak成员方法。为其定义一个__del__()魔术方法，用于显示对象是否被删除

小结：
__del__()魔术方法也可以叫做析构方法，和__init__()正好是一对
特别注意：除了del 对象以外，当页面执行完毕后，__del__()方法也会自动被触发
'''
# 1、定义一个Person类
class Person(object):
    # 2、定义成员属性
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 3、定义成员方法
    def speak(self):
        print('i can speak Chinese')
    # 4、定义一个魔术方法
    def __del__(self):
        print('当删除对象时，__del__会自动被触发')

# 5、实例化对象
p1 = Person('唐僧', 30)
print(p1.name)
print(p1.age)
p1.speak()
# 6、删除对象，查看__del__()魔术方法是否被触发
# del p1