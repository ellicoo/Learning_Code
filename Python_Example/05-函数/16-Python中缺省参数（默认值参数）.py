'''
在Python函数中，函数定义时，可以为某些参数赋予默认值，这样在调用函数时，可以不需要对默认值参数传参操作。
基本语法：
def func(a, b, c=1000):
    pass

func(10, 100)
func(10, 100, 999)

案例：定义一个func函数，拥有name、age、sex性别

注意：在定义默认值参数时，默认值参数必须放置在普通参数的右边
'''
def func(name, age, sex='male'):
    print(name)
    print(age)
    print(sex)

# 调用函数，不传默认值参数
func('Tom', 23)
# 调用函数，传递默认值参数，传递的参数会覆盖默认值
func('Rose', 24, 'female')