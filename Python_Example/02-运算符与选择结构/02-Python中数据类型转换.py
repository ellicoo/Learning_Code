'''
在Python代码中，我们可以使用如下方式实现类型转换：
int() => 把其他类型的数据转换为int整数类型
float() => 把其他类型的数据转换为float浮点类型
str() => 把其他类型的数据转换为str字符串类型

eval() => 仅作为字符串类型转换方法，把字符串两边的引号去掉，返回原数据类型
'''
# 1、字符串转数字类型
str1 = '10'
num1 = int(str1)
print(num1)
print(type(num1))

# 2、float类型转int类型
num2 = 9.69
newnum = int(num2)
print(newnum)
print(type(newnum))

# 3、把int转换为字符串类型
num3 = 6.94
str3 = str(num3)
print(str3)
print(type(str3))

# 注意：类型转换时，原数据必须是可以转换的有效数据，否则转换失败
str4 = '7.88'
num4 = int(str4)
print(num4)