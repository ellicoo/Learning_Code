'''
在Python代码中，*args与**kwargs不仅可以接收位置参数+关键词参数，其实他们还可以用于接收容器类型的数据。
*args：可以接收列表、元组等类型的数据
**kwargs：还可以用于接收字典类型的数据

list1列表
dict2字典

def func(*args, **kwargs):
    pass

func(*list1, **dict2)

案例：有一个列表，list1=[1,2,3]，dict2={'a':4, 'b':5}，要求编写一个函数，求所有元素的累加结果
1+2+3+4+5
'''
# 1、封装一个函数
def func(*args, **kwargs):
    sum = 0
    for i in args:
        sum += i
    for v in kwargs.values():
        sum += v
    return sum

# 2、定义list列表与dict字典
list1 = [1, 2, 3]
dict2 = {'a':4, 'b':5}
# 3、调用函数获取最终结果
print(func(*list1, **dict2))
