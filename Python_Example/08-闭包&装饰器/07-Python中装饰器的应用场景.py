'''
案例：使用装饰器获取程序的执行时间
'''
# 导入time模块
import time
# 编写装饰器 => ① 有嵌套 ② 有引用 ③ 有返回
def get_time(fn):
    def inner():
        begin = time.time()
        # 执行函数 => fn == func，代表要修饰函数
        fn()
        end = time.time()
        print(f'整个程序执行一共花费{end - begin}s！')
    return inner


@get_time
def func():
    list1 = []
    for i in range(1000000):
        list1.append(i)
    return list1

func()