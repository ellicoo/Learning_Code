'''
案例：获取一个列表，包含0-10之间所有的偶数 => [0, 2, 4, 6, 8]
'''
# 原生代码
list1 = []
for i in range(10):
    if i % 2 == 0:
        list1.append(i)
print(list1)

# 推导式简化for...if结构
list2 = [i for i in range(10) if i % 2 == 0]
print(list2)