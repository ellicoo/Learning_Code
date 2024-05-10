'''
简单来说，多层继承就是子类继承父类，父类在继承父类的情况。
好处：底层还是单继承，但是针对最底层的子类不仅可以拥有父类中的公共属性和公共方法，也可以拥有父类中父类的公共属性和公共方法
基本语法：
class C(object):
    pass

class B(C):
    pass

class A(B):
    pass
多层继承：实际还是单继承 => A => B => C
A不仅拥有B的所有公共属性和公共方法，还会拥有C中的所有公共属性和公共方法

面试中：Python中多继承与Java/Python中的单继承中的多层继承有何区别？
相同点：都可以实现继承关系
不同点：
多继承：属于Python、C++独有特性，一个类同时继承自多个类中的公共属性和公共方法
多层继承：本质还是单继承，只不过随着继承的传递特性，让A可以拥有B和C中的所有公共属性和方法
'''
class C(object):
    def func1(self):
        print('func1')

class B(C):
    def func2(self):
        print('func2')

class A(B):
    pass

a = A()
a.func1()
a.func2()