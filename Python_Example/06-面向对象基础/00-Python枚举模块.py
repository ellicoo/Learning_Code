"""
python枚举类
"""
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# 访问枚举成员
print(Color.RED)
print(Color.GREEN.name)
print(Color.BLUE.value)

# 遍历枚举
for color in Color:
    print(color)
