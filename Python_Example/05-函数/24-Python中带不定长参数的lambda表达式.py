'''
*args：包裹位置参数，只能接收位置参数，args返回元组类型的数据
**kwargs：包裹关键词参数，只能接收关键词参数，kwargs返回字典类型的数据
但是args & kwargs无法在lambda中混用
'''
func1 = lambda *args: args
print(func1(10, 20, 30))

func2 = lambda **kwargs: kwargs
print(func2(name='Tom', age=23, mobile='10086'))

print((lambda **args: args)(**{'a': 1, 'b': 2}))
