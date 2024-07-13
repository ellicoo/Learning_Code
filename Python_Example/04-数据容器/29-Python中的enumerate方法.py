"""
enumerate 在需要获取元素及其索引的场景中非常有用，例如在处理列表、元组或其他可迭代对象时：
"""
# 示例列表
fruits = ['apple', 'banana', 'cherry']

# 使用 enumerate 迭代
for index, value in enumerate(fruits):
    print(index, value)

print("-----------")
# 使用 enumerate 并指定起始索引
for index, value in enumerate(fruits, start=1):
    print(index, value)



print('------------')
names = ['Alice', 'Bob', 'Charlie']

for index, name in enumerate(names):
    print(f"Index {index}: {name}")
