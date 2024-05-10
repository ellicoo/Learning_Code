'''
异常的演示：
① 计算异常
② 文件不存在异常
'''
# num = int(input('请输入要进行计算的数字：'))
# print(10 / num)  # ZeroDivisionError: division by zero

f = open('bigdata.txt', 'r')  # FileNotFoundError: [Errno 2] No such file or directory: 'bigdata.txt'
content = f.read()
print(content)
f.close()
