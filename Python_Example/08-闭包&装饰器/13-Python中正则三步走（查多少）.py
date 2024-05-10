'''
* : 0次或多次
+ : 1次或多次
? : 要么为0，要么为1
{m} : 匹配前一个字符连续出现m次
{m,n} : 最少m次，最多n次
'''
import re
# 1. 匹配所有非数字字符 => \D
# result = re.match('\D*', 'abcdef1234')
# 2. 匹配所有数字字符，最少匹配1个
# result = re.match('[0-9]+', '12abcdef')
# 3. 匹配字符1或字符10
# result = re.match('10?', '10abcdef')  # 0?代表0字符可以不出现也可以只出现1次
# 4. 匹配连续3位数字
# result = re.match('\d{3}', '6667abc')
# 5. 匹配3位~5位连续数字
result = re.match('\d{3,5}', '666789abc')
print(result.group())
