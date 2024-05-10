'''
在Python代码中，没有特别的说明，则建议函数执行完毕后，把函数的最终结果通过return交还给函数的调用位置。
深入研究一下return返回值

问题1：在一个函数中，是否可以同时拥有多个return返回值？
答：可以，但是默认情况下只会返回第一个return结果，以为当函数执行到return时，系统会自动返回结果并终止整个函数执行

问题2：在其他编程语言中，一个函数默认只能返回一个结果；Python中是否可以同时返回多个结果？
答：可以，其是以元组方式返回的
'''
def func1():
    return 1
    return 2

# 调用func1函数
print(func1())  # 打印func1函数的return返回结果


def func2():
    return 1, 10, 100

# 调用func2函数
print(func2())
print(type(func2()))