'''
逻辑运算符
and ：逻辑与，两边必须同时为真，则最终结果为True；反之都为False
or ：逻辑或，只要有一边结果为真，则最终结果为True；反之都为False
not ：逻辑非，取反，真的就是假的；假的就是真的

逻辑与：and
逻辑或：or

相亲
情景一：女方条件较好，要求男方必须要有车且有房，牵手成功；反之，牵手失败！
有车 and 有房，牵手成功

情景二：女方条件一般，要求男方有车或者有房，牵手成功；反之，牵手失败！
有车 or 有房，牵手成功
'''
print(5 > 3 and 3 > 2)  # True and True => True
print(5 > 3 and 3 < 2)  # True and False => False

print(5 > 3 or 3 < 2)  # True or False => True

print(not (5 > 3))  # not True => False
