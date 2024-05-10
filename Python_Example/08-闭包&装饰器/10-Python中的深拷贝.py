'''
深拷贝是指创建一个新对象，并且这个新对象与原始对象没有任何关联。也就是说，在新对象中，原始对象中的所有元素都被复制到了新的内存地址中。如果原始对象中的元素发生了变化，那么新对象中的元素不会受到影响。
基本语法：
import copy
copy.deepcopy()
'''
import copy

list1 = [1, 2, [3, 4]]
list2 = copy.deepcopy(list1)

print(id(list1[2]))
print(id(list2[2]))

list1[2][0] = 5
print(list2)