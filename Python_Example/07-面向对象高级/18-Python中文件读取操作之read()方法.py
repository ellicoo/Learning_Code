'''
文件操作三步走
第一步：打开文件
第二步：读写文件
第三步：关闭文件
'''
f = open('python.txt', 'r', encoding='utf-8')
# content = f.read()  # 读取文件中的所有内容
content = f.read(3)   # 字符长度读取文件内容
f.close()
# 打印文件内容
print(content)