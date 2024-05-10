'''
重写？就是覆盖操作，当子类拥有与父类相同属性或方法时，则子类中的成员会覆盖父类中的成员。
基本语法：
class B(object):
    def speak():
        ...

class A(B):
    def speak():
        重写操作

a = A()
a.speak()  # 默认会访问子类中的成员方法

案例：定义一个Animal父类，父类中拥有name和age属性，以及call()方法 => 叫声，编写两个子类：Dog类，Cat类
让其自动继承Animal父类，让子类重写父类中的call()方法。

小结：
子类中包含了与父类相同的属性和方法时，则出现重写操作。
如果子类只是继承父类中的所有属性和方法是没有意义的，子类也应该拥有属于自己的属性和方法（扩展 => 重写）
'''
# 1、定义一个Animal父类
class Animal(object):
    # 2、为其定义公共属性
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 3、为其定义公共方法
    def call(self):
        print('i can call!')

# 4、编写一个Dog子类，自动继承Animal父类
class Dog(Animal):
    # 重写父类中的call方法
    def call(self):
        print('i can wang wang wang!')

# 5、编写一个Cat子类，自动继承Animal父类
class Cat(Animal):
    # 重写父类中的call方法
    def call(self):
        print('i can miao miao miao!')

# 6、实例化对象
d1 = Dog('史努比', 3)
d1.call()

c1 = Cat('加菲猫', 2)
c1.call()