'''
全局变量作用范围：探讨一下全局变量在哪里可以使用，在哪里不可以使用？
结论：
全局变量是可以在全局作用域中访问的，也可以在全局作用域中访问
'''
# 1、定义一个全局变量
num1 = 10

# 2、定义一个函数
def func():
    # 局部作用域
    print(num1)

# 3、探讨：在全局作用域中是否可以访问全局变量
# print(num1)
# 调用函数
func()