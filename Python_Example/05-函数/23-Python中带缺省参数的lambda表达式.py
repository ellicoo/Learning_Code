'''
lambda表达式不仅可以带普通参数，也可以带缺省参数
'''
# 原生函数
def func1(a, b, c=1000):
    return a + b + c

print(func1(10, 100))
# ---------------------------------------
# 使用lambda表达式实现以上代码
func2 = lambda a, b, c=1000:a + b + c
print(func2(10, 100))