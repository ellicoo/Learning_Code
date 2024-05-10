'''
lambda表达式，基本语法：
lambda 参数:返回值
'''


# 原生函数
def func1(num1, num2):
    return num1 + num2


print(func1(10, 20))

# ---------------------------------
# lambda表达式
func2 = lambda num1, num2: num1 + num2
print(func2(10, 20))
