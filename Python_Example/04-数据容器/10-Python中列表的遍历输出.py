'''
在Python代码中，列表也是容器的一种，其内部往往有多个元素。所谓的列表遍历就是把列表中的元素一个一个打印出来
while循环打印
for循环打印（推荐） => 因为for循环就是为容器遍历而生的

len()方法：可以获取字符串、列表等容器的长度或元素个数
'''
# 方案一
persons = ['刘备', '关羽', '张飞']
i = 0
while i < len(persons):  # 0 < 3 => 0 1 2
    print(persons[i])    # persons[0]、persons[1]、persons[2]
    i += 1

print('-' * 80)

# 方案二
persons = ['刘备', '关羽', '张飞']
for i in persons:
    print(i)