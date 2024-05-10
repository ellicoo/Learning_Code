'''
通用装饰器不仅可以装饰普通函数，还可以装饰可有参数以及也有返回值的一些函数。
基本语法：
① 有嵌套
② 有不定长参数
③ 有引用
④ 有返回值
⑤ 有返回，外层函数返回内层函数地址
案例：编写一个logging函数，用于装饰其他函数，从而生成日志信息
'''


# 编写一个通用装饰器
def logging(fn):
    def inner(*args, **kwargs):
        # 添加一个日志功能
        print('---Log：正在努力进行计算---')
        return fn(*args, **kwargs)

    return inner


@logging
def sum_num(num1, num2):
    return num1 + num2


@logging
def sub_num(num1, num2):
    return num1 - num2


print(sum_num(10, 20))
print(sub_num(20, 10))
