# num = [1, 2, 3, 4, 5, 6, 7, 8, ]
#
# print(num[::-1])
# print(num[:])


list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# 使用 zip 函数将两个列表打包成元组的迭代器--zip是一个迭代器，一旦转list则，指针将到两个数组的末尾。所以只能进行一次类型转换，否则第二次将空
# 即 进行了list(zipped)成功，但是ldict(zipped)则为空结果
zipped = zip(list1, list2)
my_list = list(zipped)
print(my_list)

print('-----------------')
# my_dict = {'a': 1, 'b': 2, 'c': 3}
my_dict = dict(zipped)
print(my_dict)

print('-----------测试---------')
luoma_nums = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
int_nums = [1, 5, 10, 50, 100, 500, 1000]
luoma_nums_mapping_int_nums = dict(zip(luoma_nums, int_nums))
print(luoma_nums_mapping_int_nums)

print('-------split测试------')
# s = "III"
# str= split(s,"")
# print(str)

# nums = [1]*3
# print(nums)
# n = 10
# for i in range(n - 1, 0, -1):
#     print(i)

# def get_digits(number):
#     # 获取个位数
#     ones = number % 10
#
#     # 获取十位数
#     tens = (number // 10) % 10
#
#     # 获取百位数
#     hundreds = (number // 100) % 10
#
#     # 获取千位数
#     thousands = (number // 1000) % 10
#
#     return ones, tens, hundreds, thousands
#
#
# # 示例
# number = 1234
# ones, tens, hundreds, thousands = get_digits(number)
# print("个位数:", ones)  # 输出: 4
# print("十位数:", tens)  # 输出: 3
# print("百位数:", hundreds)  # 输出: 2
# print("千位数:", thousands)  # 输出: 1

# strs  = ["flower","flow","flight"]
strs = ["dog", "racecar", "car", "xx"]


for tup in zip(*strs):
    print(tup)
    print(tup[0])

# s1 = min(strs)
# s2 = max(strs)
#
# print(s1)
#
#
# map_list = dict(zip(*strs))
# long_count=0
# for i in map_list:
#
#     print(i)
