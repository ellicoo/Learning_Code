import re

def test_pattern(pattern, test_data):
    for string in test_data:
        if re.match(pattern, string):
            print(f"'{string}' 匹配模式")
        else:
            print(f"'{string}' 不匹配模式")

pattern = r'f-(?:[1-6]?[0-9]|70)\b'
test_data = [
    "f-1",     # 匹配
    "f-50",    # 匹配
    "f-70",    # 匹配
    "f-71",    # 不匹配
    "f-111",   # 不匹配
    "f-0",     # 不匹配
    "f-100",   # 不匹配
    "f-10",    # 匹配
    "f-69",    # 匹配
    "f-80",    # 不匹配
    "f-7",     # 匹配
    "f-5",     # 匹配
    "f-6",     # 匹配
]
data = ["0f","1f","70f","71f","72f","f0","f1","f70","f71","f72","f100","f111","f-70","f-111"]
# test_pattern(pattern, test_data)
test_pattern(pattern, data)
