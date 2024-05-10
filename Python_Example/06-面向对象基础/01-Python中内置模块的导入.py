'''
import方式实现模块导入
基本语法：
import 模块名称
或
import 模块名称1, 模块名称2, ...
或
import 模块名称 as 自定义模块名称

注意：模块使用import导入完成后，必须通过如下方式进行调用
模块名称.方法()

案例1：使用import导入random随机模块
案例2：同时导入多个模块，如os模块，random模块
案例3：导入模块式，可以为其重命名

注意：通过import方式导入的模块，只能通过模块名称.方法()来进行调用！
'''
# import random
# print(random.randint(1, 10))

# import os, random
# os.mkdir('images')  # 在当前项目目录中创建一个images文件夹
# print(random.randint(1, 10))  # 生成1-10之间的随机整数

import random as rd  # 导入模块时，自动为其定义别名
print(rd.randint(1, 10))  # 别名.方法()