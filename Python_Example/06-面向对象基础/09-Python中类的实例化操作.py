'''
要想获得对象，必须要进行类的实例化操作。在其他编程语言中，都是通过new来实例化产生对象。
而Python中代码比较简单，基本语法：
    对象 = 类名()

案例：定义一个Person类，拥有两个方法eat与drink，对其实例化操作，生成对象p1
'''
# 1、定义一个Person类
class Person(object):
    # 定义公共方法
    def eat(self):
        print('我喜欢吃零食')
    def drink(self):
        print('我喜欢喝果汁')

# 2、实例化Person类，生成对象p1
p1 = Person()
print(p1)
print(type(p1))

p2 = Person()
print(p2)