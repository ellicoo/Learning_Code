'''
在Python函数中，不定长参数也可以混合在一起使用.
def func(*args, **kwargs):
    pass

以上函数既可以接收位置参数，也可以接收关键词参数
注意：如何不定长参数混合在一起使用，则*args必须放在左边，**kwargs必须放在右边
'''
# 1、定义一个func函数
def func(*args, **kwargs):
    print(args)
    print(kwargs)


# 2、调用func函数
func(1, 2, 3, a=4, b=5)
