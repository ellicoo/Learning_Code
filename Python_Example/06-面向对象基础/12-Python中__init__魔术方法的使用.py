'''
__init__()：初始化方法（在其他编程语言中，也称之为叫做构造函数），在实际工作中，这个方法有两方面作用：
① 可以用于对对象进行初始化操作，比如为其赋予相关属性
② 还可以用于实现业务初始化，比如文件操作中，可以用于打开文件；数据库操作中，可以用于连接数据库
---------------------------------------------------------------------------------
在什么情况下被触发：__init__()方法会在类的实例化时（生成对象时），会自动被触发

案例：定义一个Person类，为其赋予eat与drink成员方法，在为其添加name和age两个成员属性（类中定义）

小结：在实际工作中，我们可以通过__init__()魔术方法实现初始化操作！！！
'''
# 1、定义一个Person类
class Person(object):
    # 2、为其定义成员属性
    def __init__(self, name, age):
        # 对象.name = 传递过来的参数值
        # 对象.age = 传递过来的参数值
        self.name = name
        self.age = age

    # 3、为其定义成员方法
    def eat(self):
        print('我喜欢吃零食')
    def drink(self):
        print('我喜欢喝果汁')

# 4、实例化对象
p1 = Person('Tom', 23)
# 5、调用对象属性及方法
print(p1.name)
print(p1.age)
p1.eat()
p1.drink()

# 扩展一下：如果创建一个p2对象，其也会自动拥有name和age属性
p2 = Person('Rose', 24)
print(p2.name)
print(p2.age)