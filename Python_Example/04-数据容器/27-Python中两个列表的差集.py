list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

set1 = set(list1)
set2 = set(list2)

# 存在列表1但不存在列表2中的元素
difference = set1 - set2


difference_list = list(difference)

print(difference_list)