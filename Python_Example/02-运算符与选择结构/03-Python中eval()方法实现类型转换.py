'''

在Python代码中，我们可以使用如下方式实现类型转换：
int() => 把其他类型的数据转换为int整数类型
float() => 把其他类型的数据转换为float浮点类型
str() => 把其他类型的数据转换为str字符串类型

eval() => 仅作为字符串类型转换方法，把字符串两边的引号去掉，返回原数据类型 => 只能实现对字符串的类型转换，转换为原数据类型

'''

str1 = '10'  # str字符串
num1 = eval(str1)  # 去掉引号，返回结果为10，把10赋值给num1，所以num1就是int类型
print(num1)
print(type(num1))

str2 = '9.88'  # str字符串
num2 = eval(str2)  # 去掉引号，返回结果为9.88，把9.88赋值给num2，所以num2就是float类型
print(num2)
print(type(num2))

str3 = 'abc123'
num3 = eval(str3)  # 报错--只能实现对字符串的类型转换，转换为原数据类型
print(num3)
