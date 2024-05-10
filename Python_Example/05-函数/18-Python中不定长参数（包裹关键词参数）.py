'''
就是在函数调用时不一定传递多少个参数（也存在不传参的情况），这种情况，如何定义函数的参数呢？
答：不定长参数

不定长参数有两种方式：
def func(*args) ：包裹位置不定长参数
def func(**kwargs) ：keyword，包裹关键词不定长参数

相同点：都可以用于接收不定长参数
不同点：
*args只能接收位置参数
**kwargs只能接收关键词参数
返回值类型也有所不同
*args：args是最终参数，以元组方式返回
**kwargs：kwargs是最终参数，以字典方式返回
'''
def func(**kwargs):
    print(kwargs)
    print(type(kwargs))

# 不传递任何参数
func()
# 只传递1个参数
func(name='Tom')
# 也可以同时传递多个参数
func(name='Tom', age=23, mobile='10086')