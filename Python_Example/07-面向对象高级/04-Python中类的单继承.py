'''
继承基本语法：
class B(object):
    pass

class A(B):
    pass

案例：定义一个Person父类，包含name、age公共属性以及speak()公共方法；在这个基础上在定义一个Student子类，
自动继承Person类中的所有公共属性与公共方法

小结：
所谓的继承就是子类自动继承父类中的公共属性和公共方法
作用：实现代码重用，减少重复性代码编写
注意：子类只能继承父类中的公共属性和公共方法
'''
# 1、定义一个父类Person类
class Person(object):
    # 2、定义公共属性
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 3、定义公共方法
    def speak(self):
        print(f'我的名字叫做{self.name}，今年{self.age}岁了！')

# 4、定义一个Student子类，自动继承Person父类
class Student(Person):
    pass
# 5、验证，由Student类实例化的对象是否会自动拥有父类中的公共属性和公共方法
s1 = Student('小明', 23)
# 访问s1对象的属性和方法
print(s1.name)
print(s1.age)
s1.speak()