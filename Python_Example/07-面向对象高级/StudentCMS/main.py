'''
项目主文件：在Python项目开发中，主入口程序都是通过main.py定义
在Java/C++等程序中，程序的启动，比如放在入口函数中 => bot_userprofile_dev()函数实现
但是在Python代码中，没有main函数，如何定义程序的启动入口（文件真正的启动函数），我们可以通过如下代码实现：
if __name__ == '__main__':
    实例化，以及启动操作
在以后的开发中，核心实例化、初始化启动等等都必须放入以上代码中，这就相当于程序的入口
扩展：PyCharm中有一个快捷键 => Ctrl + 鼠标左键
'''
# 导入学生管理系统类
from StudentCMS import StudentCMS
if __name__ == '__main__':
    # 1、实例化学生管理系统，生成一个对象
    cms = StudentCMS()
    # 2、调用对象中的启动方法，初始化相关操作
    cms.start()