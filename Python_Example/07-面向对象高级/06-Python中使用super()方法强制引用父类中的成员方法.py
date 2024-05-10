'''
基本语法：
class B(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

class A(B):
    def __init__(self, name, age, mobile):
        # 继承父类中的name和age属性
        super().__init__(name, age)  # 强制引用父类中的公共属性或公共方法
        # 重写，让子类对象自动拥有mobile属性
        self.mobile = mobile

a = A()
a.name/a.age/a.mobile

# 案例：定义一个父类Car，父类拥有brand、color颜色等属性，定义一个子类ElectricCar，自动继承父类中的所有公共属性和公共方法；在继承的同时，扩展一个
battery属性（电池容量）信息。

小结：
在类的继承中，如果子类重写了父类中的公共属性和公共方法，我们也可以通过super().属性或super().方法()方式强制引用父类中的公共属性和公共方法！
'''
# 1、定义一个Car父类
class Car(object):
    # 2、定义魔术方法__init__()为其初始化brand和color属性
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color
    # 3、定义一个run方法
    def run(self):
        print('i can run!')

# 4、定义一个ElectricCar子类，自动继承Car父类
class ElectricCar(Car):
    # 5、重写父类中的__init__()方法，重写的同时，还需要延伸自己的属性battery电池容量
    def __init__(self, brand, color, battery):
        # 强制引用父类中的brand和color属性
        super().__init__(brand, color)
        self.battery = battery

# 6、实例化ElectricCar，获取相关属性和方法
tesla = ElectricCar('特斯拉', '蓝色', '10w')
print(tesla.brand)
print(tesla.color)
print(tesla.battery)