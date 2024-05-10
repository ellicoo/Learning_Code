'''
函数嵌套：所谓的嵌套就是在一个函数中又调用了另外一个函数
基本语法：
def 函数1():
    ...
    函数2()

探讨函数嵌套的执行流程，疑问：当以下代码执行到func2()函数调用位置，则后续如何执行？
A. 先把func1函数中未执行的代码执行完毕，在执行func2函数中的代码
B. 执行到func2()函数位置，则跳转到func2函数内部，执行其内部代码；执行完毕后，返回继续执行func1中未执行代码

答案：B
'''
def func2():
    print('func2 start')
    print('---- func2 ----')
    print('func2 end')

def func1():
    print('func1 start')
    # 在一个函数中又调用了另外一个函数
    func2()
    print('func1 end')

# 调用func1()函数
func1()