'''
用户登录案例
① 输入用户名和密码
② 判断用户名和密码是否正确（username='admin'，password='admin888'）
③ 登录仅有三次机会，超过3次会报错
'''
# 尝试登录次数，默认为0
trycount = 0
# 1、编写一个循环，循环3次
for i in range(3):
    # 每循环一次，trycount += 1
    trycount += 1
    # 2、提示用户输入账号和密码
    username = input('请输入您的账号：')
    password = input('请输入您的密码：')
    # 3、判断一下用户名是否正确
    if username == 'admin':
        # 4、判断一下密码是否正确
        if password == 'admin888':
            print('登录成功')
            break
        else:
            print('很抱歉，您的密码有误，请重新输入')
            print(f'您还剩余{3 - trycount}次登录机会')
    else:
        print('很抱歉，您的账号有误，请重新输入')
        print(f'您还剩余{3 - trycount}次登录机会')
