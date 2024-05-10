'''
在实际工作中，很少在类的外部直接对类的内部的属性进行访问。面向对象强调封装特性，属性的访问都是通过方法实现的。
同样的，类中的类属性，也需要依赖类方法实现访问。
基本语法：
class 类名():
    属性 = 属性的值  # 类属性
    @classmethod
    def get_count(cls):
        print(cls.属性)
'''
# 1、定义一个Tool工具类
class Tool(object):
    # 2、定义一个类属性，用于记录生成对象的数量
    count = 0
    # 3、定义一个__init__()魔术方法，用于初始化对象属性
    def __init__(self, name):
        self.name = name
        Tool.count += 1

    # 4、定义一个类方法
    @classmethod
    def get_count(cls):
        print(f'由Tool工具类共实例化{cls.count}个对象')

# 5、实例化对象
t1 = Tool('斧头')
t2 = Tool('榔头')
t3 = Tool('锤子')

# 6、调用类方法打印输出类属性信息
Tool.get_count()