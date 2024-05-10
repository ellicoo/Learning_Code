'''
自定义模块的测试方法，要求：
测试功能只在my_module2（当前页面）有效，当其他页面调用此模块时，测试功能自动失效！
要想实现以上功能，可以使用
if __name__ == '__main__'方式实现
__name__：魔术变量（在Python中拥有特殊含义的变量）
__name__：在当前模块中运行，其值为__main__字符串，如果其他页面调用此模块，则__name__，其值为当前模块的名称
基于这个特性，就可以使用if __name__ == '__main__'实现自定义模块中的功能测试
'''
# 1、定义函数
def func(name, age, mobile):
    print(name)
    print(age)
    print(mobile)


# 2、测试函数（调用函数）
if __name__ == '__main__':
    func('Tom', 23, '10086')