'''
readlines()：一次性读取文件所有内容，适合小文件读取
readline()：每次只读取文件的一行，然后移动文件指针到下一行，和read(num)类似都适合大文件读取
'''
# 1、打开文件
f = open('python.txt', 'r', encoding='utf-8')
# 2、读写文件
while True:
    content = f.readline()
    if not content:  # 文件读取结束，content返回一个空 => if conent => False
        break
    print(content, end='')
# 3、关闭文件
f.close()

'''
小结：
在Python代码中，读取文件一共有3种方法
f.read(num) ：读取指定num长度的文件内容，如果不指定num，默认读取所有文件内容
f.readlines() ：一次性读取所有文件内容，默认以列表方式范围，每一个元素就是源文件中的一行
f.readline() ：每次读取文件的一行

read()与readline()方法既适合大文件也适合小文件
readlines()只适合读取小文件
'''