'''
isdigit()：字符串.isdigit()，判断字符串是否为纯数字组成
'''
password = input('请输入您的6位银行卡密码：')
if password.isdigit():
    print('验证通过，继续往下执行')
else:
    print('验证不通过，请重新输入')