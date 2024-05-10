import re
def replace_string(input_str):
    if input_str in ['Japan', 'ROOM']:
        return input_str

    # Regular expression pattern:
    # (?=.*[a-zA-Z])  # At least one letter
    # ^[a-zA-Z0-9*]+$  # Only letters, numbers, and asterisks
    #
    # pattern1 = r'^(b[1-5]f|(?:[1-9]|[1-6][0-9]|70)f|[1-5]b|b[1-5]|f(?:[1-9]|[1-6][0-9]|70)|\d+(?![bf])[a-zA-Z])$'
    #
    # if re.match(pattern1, input_str, re.IGNORECASE):
    #     return input_str

    # 匹配数字开头，除了'b'和'f'外的任意字符，不进行转换htn
    pattern1 =  r'^\d+(?![bf])[a-zA-Z]$'
    # 匹配除了'b'和'f'以外的任意字母开头+数字结尾，数字大于等于1, 存疑
    pattern2 = r'^(?![bf])[a-zA-Z]+\d{1,}$'

    # 匹配以'b'开头/结尾,b+数字+f结尾，数字大于等于1小于5，不进行转换
    pattern3 = r'^(b[1-5])$'
    pattern4 = r'^([1-5]b)$'
    pattern5 = r'^b[1-5]f$'
    # 匹配以'f'开头/结尾，数字大于等于1小于等于70，不进行转换
    pattern6 = r'^f(?:[1-9]|[1-6][0-9]|70)$'
    pattern7 = r'^(?:[1-9]|[1-6][0-9]|70)f$'

    # 匹配以'f-'开头，数字大于等于1小于等于70，不进行转换
    # pattern8 = r'^(?![bf])[a-zA-Z]-(?:[1-9]|[1-6][0-9]|70)$'
    # 匹配任意字母开头+-+数字结尾  存疑
    pattern8= r'^[a-zA-Z]-\d+$'

    # 检查是否匹配到不需要转换的模式
    if re.match(pattern1, input_str, re.IGNORECASE) or re.match(pattern2, input_str, re.IGNORECASE) or re.match(
            pattern3, input_str, re.IGNORECASE) \
            or re.match(pattern4, input_str, re.IGNORECASE) or re.match(pattern5, input_str, re.IGNORECASE) or re.match(
        pattern6, input_str, re.IGNORECASE) \
            or re.match(pattern7, input_str, re.IGNORECASE) or re.match(pattern8, input_str, re.IGNORECASE):
        return input_str

    pattern9 = r'(?=.*[a-zA-Z])^[a-zA-Z0-9*]+$'
    #
    # # re.sub replaces the string that matches the pattern with 'RANDOM'
    return re.sub(pattern9, 'RANDOM', input_str)


def convert_list_to_str_for_bert(input_list):
    return [replace_string(each) for each in input_list]


# 测试数据
# 函数测试

# 测试f  OK
data = ["3E","A-0","a-1f","5a","0f","1F","70f","71f","72f","f0","f1","F70","f71","f72","f100","f111","f-70","f-71","a-11","a3"]

# 测试b+数字+f OK
# data = ["B0F", "B1F", "B5F", "B6F", "B123F"]
#
# 测试b OK
# data = ["0B","1B","5B","6B","10B","11B","12B","B0","B1","B5","B6","B10","B11","B12"]

# 测试匹配任意一个英文+字母
# data = ["a5", "f72","b10", "c100", "a-5","d-5","d-100","f-100"]
# data = ["a-5", "b-2", "c-9", "d-0","a*5","a6", "b7", "c8", "d9"]

# data = ["1-10-3", "g3", "ビル", "1f", "新宿区"]
# data = ["2-3-23", "三軒ビル", "b1fc2remit", "新宿"]

result = convert_list_to_str_for_bert(data)
print(result)
