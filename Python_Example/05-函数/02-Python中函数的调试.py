'''
案例：封装一个函数，用于求两个参数的和
函数调试步骤三步走：
第一步：下断点（断点要打在函数的调用位置）
第二步：启动Debug调试
第三步：使用Step Into一步一步调试代码
-------------------------------------------------
Python基础语法、if、while、for循环结构都是通过Step Over
Step Into：代码一步一步向下执行，遇到函数，则跳转到函数内部，继续执行
Step Over：代码一步一步向下执行，遇到函数，不跳转到函数内部，而是直接返回函数的执行结果，继续执行
'''
# 1、定义一个函数sum_num，为其定义两个参数, num1与num2
def sum_num(num1, num2):
    return num1 + num2

# 2、调用sum_num函数，然后把函数返回值赋值给变量content
content = sum_num(10, 20)

# 3、打印输出content
print(content)
