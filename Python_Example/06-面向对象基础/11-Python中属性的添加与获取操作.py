'''
在Python代码中，我们可以在类的外部为对象添加和获取属性。
基本语法：
① 设置对象属性
    对象.属性 = 属性值
② 获取对象属性
    对象.属性

案例：定义一个Person类，为其定义eat和drink()成员方法，并在类的外部，为其对象添加name和age属性
'''
# 1、定义一个Person类
class Person(object):
    # 2、在类内部添加eat与drink()成员方法
    def eat(self):
        print('我喜欢吃零食')
    def drink(self):
        print('我喜欢喝果汁')
# 3、实例化生成对象p1
p1 = Person()
# 4、在类外部为p1对象添加name和age属性 => Tom, 23
p1.name = 'Tom'
p1.age = 23
# 5、在类外部输出对象p1的属性和方法
# 打印属性
print(p1.name)
print(p1.age)
# 打印方法
p1.eat()
p1.drink()