'''
在其他编程语言中，只有if结构才拥有else。但是在Python中比较特殊，不仅if拥有else结构。我们的循环也支持else结构
如果while...else，for...else，基本语法：
for 临时变量 in 数据容器:
    print(临时变量)
else:
    print('当循环正常结束时，系统会自动执行else中的代码')

疑问：什么情况是正常结束？什么情况是非正常结束？
答：在循环结构中，只要没有执行break语句都是正常结束；一旦执行了break关键字就代表非正常结束！

for...else中的else在实际工作中有何作用？
答：我们可以使用else实现收尾工作！！！
'''
# 正常结束
# for i in 'itheima':
#     print(i)
# else:
#     print('当循环正常结束时，系统会自动执行else中的代码')

# 非正常结束
for i in 'itheima':
    if i == 'e':
        print('遇到e则终止整个循环')
        break
    print(i)
else:
    print('当循环正常结束时，系统会自动执行else中的代码')