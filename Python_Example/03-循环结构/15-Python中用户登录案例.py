'''
用户登录案例
① 输入用户名和密码
② 判断用户名和密码是否正确（username='admin'，password='admin888'）
③ 登录仅有三次机会，超过3次会报错
'''
# 1、编写一个循环，循环3次
for i in range(3):
    # 2、提示用户输入用户名 + 密码
    username = input('请输入您的账号：')
    password = input('请输入您的密码：')
    # 3、判断用户名和密码是否正确，正确则退出循环，否则进入下一次循环
    if username == 'admin' and password == 'admin888':
        print('登录成功')
        break
    else:
        print('很抱歉，用户名或密码错误！')
else:
    print('循环已结束，账号锁定5分钟！')