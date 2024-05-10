'''
很多小伙伴会比较好奇 => 新式类中为什么要保留一个小括号，答：为了实现继承操作
class B():
    ...

class A(B):  # 这句话的含义代表让A这个子类继承B中的属性和方法
    ...

目前为止，我们还没有学到继承操作，这个小括号中应该写什么呢？
答：可以写object，object是一个顶级类，也可以理解是所有类的父类，所有Python类都会自动继承object中的属性和方法
'''
class Person(object):
    # 定义一个eat方法()
    def eat(self):
        print('我喜欢吃零食')
    # 定义一个drink方法()
    def drink(self):
        print('我喜欢喝果汁')