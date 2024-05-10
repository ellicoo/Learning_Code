'''
replace()：字符串.replace(旧关键词，新关键词)，返回替换后的内容
split()：字符串.split('切割符号')，如apple-banana-orange => ['apple', 'banana', 'orange']
join()：字符串.join(列表容器)，把列表容器中的元素通过字符串拼接成一个新字符串
'''
str1 = 'hello linux'
print(str1.replace('linux', 'python'))

str2 = 'apple-banana-orange'
print(str2.split('-'))

list3 = ['apple', 'banana', 'orange']
str3 = '-'.join(list3)
print(str3)