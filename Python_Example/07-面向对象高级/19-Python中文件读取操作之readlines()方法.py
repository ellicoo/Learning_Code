'''
readlines()：一次性读取文件的所有内容，以行为单位，每一行就是列表中的一个元素
'''
f = open('python.txt', 'r', encoding='utf-8')
lines = f.readlines()
f.close()

# 打印文件内容
# print(lines)

# 去掉换行符
for content in lines:
    print(content, end='')