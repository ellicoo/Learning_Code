'''
多态的使用条件：
① 有继承
② 子类必须重写父类中的公共方法（相同方法）
③ 定义一个函数，随着传入子类对象的不同，可以返回不同的执行结果
'''
class Fruit(object):
    def makejuice(self):
        print('i can make juice!')

class Apple(Fruit):
    def makejuice(self):
        print('i can make apple juice!')

class Banana(Fruit):
    def makejuice(self):
        print('i can make banana juice!')

# 定义一个公共方法（公共接口）
def func(obj):
    obj.makejuice()

# 同一个func函数，随着传入子类对象的不同，返回不同的执行结果
apple = Apple()
banana = Banana()

func(apple)
func(banana)

# 特别说明：自己编写多态情况较少，但是在Python底层，很多功能都是通过多态实现，比如+运算符，就是多态！
# 同一个+加号，随着两边参数的不同，功能也有所不同
# 两边是数字，则+就是加法运算，比如1+1=2
# 两边是容器类型，则+就是合并操作，比如[1,2] + [3,4] = [1, 2, 3, 4]