'''
在Python中，赋值运算符只有一个，就是等号=
基本语法：
变量名称 = 变量的值/程序表达式
=叫做赋值，其执行顺序都是从右向左执行
'''
# 1、普通变量赋值操作
name = 'Tom'
sum = 0
sum = 1 + 2 + 3
print(sum)
# 2、同时为多个变量赋值
a, b, c = 10, 20, 30
print(a)
print(b)
print(c)
# 3、多个变量赋予相同的值
num1 = num2 = 100
print(num1)
print(num2)