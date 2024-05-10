'''
在Python代码中，如果想让多个函数之间共享数据，必须把这个数据声明为全局变量。
'''
num = 10

def func1():
    global num
    num = 100
    print(num)

def func2():
    print(num)

func1()
func2()