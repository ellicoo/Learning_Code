'''
当程序出现异常，准备一个B方案 => 异常捕获
基本语法：
try:
    可能出现异常的代码
except:
    如果程序出现异常，系统会自动执行except中的代码

捕获异常并弹出异常信息
try:
    可能出现异常的代码
except Exception as e:
    print(e)  # 异常信息
    如果程序出现异常，系统会自动执行except中的代码
'''
try:
    num = int(input('请输入要进行计算的数字：'))
    print(10 / num)
except Exception as e:
    print(e)
    print('很抱歉，您的输入有误，请重新输入！')