'''
案例：求三个数的平均值
'''
# 1、定义一个函数
def avg_num(a, b, c):
    # 函数的说明文档
    '''
    返回三个数的平均值
    :param a: int, 第一个参数
    :param b: int, 第二个参数
    :param c: int, 第三个参数
    :return: 三个数的平均值
    '''
    # 2、返回三个数的平均值
    return (a + b + c) / 3


# 3、调用函数
print(avg_num(10, 20, 30))

