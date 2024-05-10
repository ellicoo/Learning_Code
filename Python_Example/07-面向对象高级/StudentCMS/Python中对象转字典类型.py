'''
在Python代码中，每一个对象都有一个__dict__魔术变量，这个变量可以把对象转换为字典类型。
'''
class A(object):
    def __init__(self, name):
        self.name = name
        self.__age = 18

a = A('Tom')
print(a.__dict__)