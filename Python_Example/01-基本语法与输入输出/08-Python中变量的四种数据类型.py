'''
在Python中，一共有7种数据类型，今天只需要记住：三大类四小类
① 数字类型
int类型：整数类型
float类型：浮点类型
② 布尔类型
bool类型
③ 字符串类型
str类型

在Python代码中，我们可以使用type(变量名称)来判断变量的数据类型
'''
# 1、定义整数类型
a = 10
print(type(a))  # int
# 2、定义浮点类型
b = 9.88
print(type(b))  # float
# 3、定义布尔类型，布尔只有两个值，要么为True（真），要么为False（假）
c = True
print(type(c))
# 4、定义字符串类型
d = 'python'
print(type(d))