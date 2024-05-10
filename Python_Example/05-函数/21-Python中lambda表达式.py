'''
lambda表达式就是匿名函数，主要用于简化业务逻辑比较简单的普通函数。
'''
# 原生函数
def func1():
    return 100

print(func1())

# ----------------------------
# lambda表达式
func2 = lambda :100
print(func2())