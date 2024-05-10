'''
基本语法：
try:
    可能出现异常的代码
except Exception as e:
    print(e)
    print('如果出现异常，则执行except中的代码')
else:
    如果try语句中没有出现异常，则执行else中的代码；反之，则不执行
finally:
    无论程序是否出现异常，finally语句中的代码都会被执行
'''
try:
    f = open('bigata.txt', 'r', encoding='utf-8')
except Exception as e:
    print(e)
    f = open('bigata.txt', 'w', encoding='utf-8')
else:
    content = f.read()
    print(content)
finally:
    f.close()
