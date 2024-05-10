'''
在定义Python新式类时，每定义一个方法，其内部都会产生一个self关键字（参数），其指向实例化对象本身
简单来说：就是谁实例化这个类，它就指向谁
说明：在实例化产生对象以后，我们可以使用对象.方法()方式实现对类中成员方法的访问

小结：在Python新式类中，成员方法中的self关键字指向实例化对象本身
'''
# 1、定义一个Person新式类
class Person(object):
    def func(self):
        print(self)

# 2、实例化Person类，生成p1对象
p1 = Person()
print(p1)  # 返回p1对象指向的内存地址
p1.func()  # 找到对象中的func()方法，并立即调用

p2 = Person()
print(p2)
p2.func()
