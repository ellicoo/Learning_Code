'''
观察以下代码，说出程序的执行结果。
注意：在函数中，函数的参数也是一个局部变量！！！
'''


# 1、有嵌套
def outer(num1):
    def inner(num2):
        # 2、有引用
        result = num1 + num2
        print(result)

    # 3、有返回
    return inner


# 调用outer函数
func = outer(1)  # 返回的是inner(num2)函数的地址和1+num2 --局部变量不会消失
func(2)  # inner函数的地址(2)，相当于给1+num2 -> 1+2=3，这个时候num1=1依然驻留在内存中
func(3)  # inner函数的地址(3)，相当于给1+num2 -> 1+3=4，这个时候num1=1依然驻留在内存中
