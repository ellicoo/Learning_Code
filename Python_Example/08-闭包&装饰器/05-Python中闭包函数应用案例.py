'''
观察以下代码，说出程序的执行结果。
global：声明全局变量
nonlocal：声明当前函数外部函数中的局部变量
'''


# 1、有嵌套
def outer():
    num1 = 1

    def inner():
        # 声明外部局部变量
        nonlocal num1
        # 2、有引用
        num1 += 1
        print(num1)

    # 3、有返回
    return inner


# 调用outer函数
func = outer()
func()  # 2
func()  # 3
func()  # 4
