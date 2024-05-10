# 1、打开文件
f = open('python.txt', 'w', encoding='utf-8')  # r读取/w清空写入/a追加写入
# 2、读写文件
f.write('Life is Short，I Study Python！')
# 3、关闭文件
f.close()

'''
小结：
文件操作一共分三步：
第一步：打开文件
第二步：读写文件
第三步：关闭文件

练习：使用Python实现对文件的写入操作，新文件名称bigdata.txt，写入内容："大数据技术栈包含Hadoop、Hive、Spark等等"，限时3分钟！
'''