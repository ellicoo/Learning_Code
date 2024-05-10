'''
+ ：合并操作
* ：复制操作
in ：判断元素是否出现在容器中，出现True；反之False
max() ：返回容器中元素的最大值
min() ：返回容器中元素的最小值
'''
# 定义两个字符串，演示拼接操作
str1 = 'hello'
str2 = 'world'
print(str1 + str2)

# 定义两个列表，演示合并操作
list1 = [1, 3, 5]
list2 = [2, 4, 6]
print(list1 + list2)

# in关键词 => Python + Web框架 = 运维系统（黑名单）
black_ips = ['10.1.1.1', '222.246.79.81', '192.168.13.200']
if '10.1.1.1' in black_ips:
    print('禁止访问')
else:
    print('正常访问')

# max()与min()返回最大值与最小值
tuple1 = (10, 20, 30)
max_value = max(tuple1)
min_value = min(tuple1)
print(f'最大值：{max_value}')
print(f'最小值：{min_value}')