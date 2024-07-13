'''
浅拷贝是指创建一个新对象，但是这个新对象只是原始对象的一个引用。也就是说，在新对象中，原始对象中的所有元素都只是引用。如果原始对象中的元素发生了变化，那么新对象中的元素也会发生变化。在Python中，可以使用copy()方法来进行浅拷贝。
浅拷贝：
两个对象内容相同，只能拷贝外层对象，内层对象和原对象是引用关系
'''

list1 = [1, 2, [3, 4]]
list2 = list1.copy()

print(id(list1[2]))
print(id(list2[2]))

# 当源对象中的元素发生改变时，由于两者指向了相同的内存地址，新对象也会随之改变
list1[2][0] = 5

print(list2)

# 注意：切片拷贝也是浅拷贝
list3 = list1[:]
print(list3)


print('----------深拷贝&浅拷贝-------------')
# 普通赋值是--浅拷贝
import copy

# 示例对象
original_list = [1, 2, [3, 4]]

# 浅拷贝
shallow_copy = original_list.copy()
# 深拷贝
deep_copy = copy.deepcopy(original_list)

# 修改原始对象
original_list[2][0] = 'modified'

print("Original List:", original_list)       # Output: Original List: [1, 2, ['modified', 4]]
print("Shallow Copy:", shallow_copy)         # Output: Shallow Copy: [1, 2, ['modified', 4]]
print("Deep Copy:", deep_copy)               # Output: Deep Copy: [1, 2, [3, 4]]

