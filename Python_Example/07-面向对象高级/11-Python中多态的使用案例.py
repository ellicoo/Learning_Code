'''
Python中多态：一个事物多种形态 => 同一个函数，随着传入子类对象的不同，可以返回不同的执行结果。
案例：定义一个Dog父类，内部有一个work公共方法。定义两个子类ArmyDog、DrugDog，继承Dog类并重写work方法；
封装一个公共方法，随着传入对象的不同，返回不同的执行结果。
'''
class Dog(object):
    def work(self):
        print('指哪打哪')

class ArmyDog(Dog):
    def work(self):
        print('追击敌人')

class DrugDog(Dog):
    def work(self):
        print('追查毒品')

class Person(object):
    # 定义一个公共方法
    def work_with_dog(self, dog):
        dog.work()

p1 = Person()

ad = ArmyDog()
dd = DrugDog()

p1.work_with_dog(ad)  # 追击敌人
p1.work_with_dog(dd)  # 追查毒品
