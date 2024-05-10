'''
Python是少数编程语言中支持多继承特性的，C++也是支持多继承的。基本语法：
class B(object):
    pass

class C(object):
    pass

class A(B, C):
    pass
    A自动继承B和C中的所有公共和公共方法

案例：定义一个油车类，定义一个电车类，定义一个混合动力汽车类，自动继承油车以及电车类中的相关公共属性和公共方法。

小结：
多继承中，子类不仅可以继承父类中的公共属性和公共方法，还必须要遵循mro继承链！！！
'''
class GasolineCar(object):
    # 定义魔术方法__init__
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color
    # 定义一个公共方法
    def run_with_gasoline(self):
        print('i can run with gasoline')

    def func(self):
        print('one')

class ElectricCar(object):
    # 定义魔术方法__init__
    def __init__(self, brand, color, battery):
        self.brand = brand
        self.color = color
        self.battery = battery
    # 定义一个公共方法
    def run_with_electric(self):
        print('i can run with electric')

class HybridCar(GasolineCar, ElectricCar):  # Hybrid => GasolineCar => ElectricCar => object
    def __init__(self, brand, color, battery):
        super(GasolineCar, self).__init__(brand, color, battery)

# 实例化子类对象
tesla = HybridCar('特斯拉', '蓝色', '81kWh')
tesla.run_with_gasoline()
tesla.run_with_electric()

# 疑问：如果两个父类中存在了相同的属性和方法，子类会自动继承哪个类中的属性和方法呢？
# 答：在Python中，默认的继承关系要遵循mro继承链 => 子类名称.__mro__或子类名称.mro()来查看继承关系
print(HybridCar.__mro__)
print(HybridCar.mro())
