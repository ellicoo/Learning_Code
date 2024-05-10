'''
在Python中，要想获得一个对象，必须先有一个类，基本语法：
class 类名():
    属性
    方法（函数）

# 案例：定义一个Person类，拥有两个方法eat()方法，drink()方法
'''
class Person():
    # 定义一个eat方法()
    def eat(self):
        print('我喜欢吃零食')
    # 定义一个drink方法()
    def drink(self):
        print('我喜欢喝果汁')