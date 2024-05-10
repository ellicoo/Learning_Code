'''
三步走：
import re
result = re.match(正则表达式, 要匹配的字符串)  # 限制，要匹配的内容必须放置在字符串的最前面，一次只能匹配1个字符，一个re.match对象
result.group()

import re
result = re.findall(正则表达式, 要匹配的字符串)  # 没有限制，一次匹配所有满足条件的结果，返回结果是列表
result

正则表达式 = 普通字符 + 元字符（表达式特有）
'''
# 案例1：从字符串中查找数字8
# import re
# result = re.findall('8', '13575666844')
# print(result)

# 案例2：从字符串中匹配所有的数字
# import re
# result = re.findall('\d', 'a1b2c3d4e5f6')
# print(result)

# 案例2：从字符串中匹配所有的非数字字符
import re
result = re.findall('\D', 'a1b2c3d4e5f6')
print(result)
