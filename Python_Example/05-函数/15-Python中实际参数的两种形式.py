'''
在Python代码中，实际参数往往有两种传递方式：① 位置传参 ② 关键词传参
def func(a, b, c):
    pass
① 位置参数：要求传递的参数与形参参数的位置必须要一一对应
func(10, 20, 30)
② 关键词参数：没有位置要求，但是传参形式必须采用参数=值方式
func(a=10, b=20, c=30)
'''
def func(name, age, mobile):
    print(name)
    print(age)
    print(mobile)

# 调用时使用位置传参
func('Tom', 23, '10086')

# 调用时使用关键词传参
func(name='Tom', age=23, mobile='10086')