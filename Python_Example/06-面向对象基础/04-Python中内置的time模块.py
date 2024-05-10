'''
在Python中，内置了一个time模块，与时间相关操作。
time.sleep(秒数) ：让程序休眠指定的秒数
time.time() ：获取当前时间时间戳（从1970-1-1格林制时间）到当前时间的秒数（精确到毫秒）

面试过程中：如何评估程序的执行时间 => time.time()来实现
'''
# 1、导入time模块
import time
# 2、程序执行之前获取一个时间戳
start_time = time.time()
# 要执行的程序
list1 = []
for i in range(1, 1000000):
    list1.append(i)
# 3、程序执行之后获取一个时间戳
end_time = time.time()
# 4、在程序运行结束，获取结束时间戳 - 开始时间戳
print(f'整个Python程序执行一共消耗了{end_time - start_time}s时间')