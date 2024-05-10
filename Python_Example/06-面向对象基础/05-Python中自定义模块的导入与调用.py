'''
不管是系统模块还是自定义模块，我们都可以通过import/from对其进行导入。
'''
# 1、导入自定义模块
from my_module1 import sum_num, sub_num
# 2、调用模块中的相关方法
print(sum_num(10, 20))
print(sub_num(20, 10))