'''
在Python的多继承中，一个子类同时继承了多个父类情况。特殊情况：如果多个父类中，存在相同的方法时，默认是按照mro继承链方式从左向右继承。问：有没有办法，指定调用某个父类中的方法。
答：也可以，通过super(父类名称, self).方法名()实现

super()方法特殊性：
实际工作中，都是通过super().func()来调用父类中的相关属性和方法。
以上程序的完整写法 => super(子类名称, self).func() => 调用子类对应的父类中的func()方法
'''
class B(object):
    def func(self):
        print('B类中的func方法')

class C(object):
    def func(self):
        print('C类中的func方法')

class A(B, C):  # A => B => C => object
    def func(self):
        # 获取C中的func方法
        super(B, self).func()

a = A()
a.func()