'''
其实，Python代码中，模块不仅可以使用import进行导入，还可以使用from方式进行导入操作。
基本语法：
from 模块名称 import *
或
from 模块名称 import 方法名
或
from 模块名称 import 方法名1, 方法名2, ...

调用方式：
使用了from导入模块中的方法以后，调用时直接写方法名称即可，不需要添加模块这个前缀。
方法名称()

# 案例1：导入os模块中的所有方法，创建一个文件夹叫做media
# 案例2：只导入我们要使用的某个方法，如导入random模块中的randint()方法
# 案例3：同时导入os模块中的多个方法
'''
# 导入模块
# from os import *
# 调用mkdir()方法
# mkdir('media')

# from random import randint
# print(randint(1, 10))

from os import mkdir, rmdir
rmdir('media')