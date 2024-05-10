'''
在Python代码中，不仅while循环有break和continue；在for循环中也可以使用break和continue。
break：终止整个循环结构
continue：中止本次循环，继而进入下一次循环
'''
# 情景一：break关键字使用
# 打印itheima中的每一个元素，遇到字符e则终止整个循环
# for i in 'itheima':
#     if i == 'e':
#         print('遇e则终止整个循环结构')
#         break
#     print(i)

# 情景二：continue关键字使用
# 打印itheima中的每一个元素，遇到字符e则跳过本次循环，继而进入下一次循环
for i in 'itheima':
    if i == 'e':
        print('遇e则跳过本次循环，继而进入下一次循环')
        continue
    print(i)