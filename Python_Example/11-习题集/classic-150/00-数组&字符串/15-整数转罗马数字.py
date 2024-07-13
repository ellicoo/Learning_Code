"""
七个不同的符号代表罗马数字，其值如下：

符号	值
I	1
V	5
X	10
L	50
C	100
D	500
M	1000
罗马数字是通过添加从最高到最低的小数位值的转换而形成的。将小数位值转换为罗马数字有以下规则：

如果该值不是以 4 或 9 开头，请选择可以从输入中减去的最大值的符号，将该符号附加到结果，减去其值，然后将其余部分转换为罗马数字。
如果该值以 4 或 9 开头，使用 减法形式，表示从以下符号中减去一个符号，例如 4 是 5 (V) 减 1 (I): IV ，9 是 10 (X) 减 1 (I)：IX。仅使用以下减法形式：4 (IV)，9 (IX)，40 (XL)，90 (XC)，400 (CD) 和 900 (CM)。
只有 10 的次方（I, X, C, M）最多可以连续附加 3 次以代表 10 的倍数。你不能多次附加 5 (V)，50 (L) 或 500 (D)。如果需要将符号附加4次，请使用 减法形式。
给定一个整数，将其转换为罗马数字。



示例 1：

输入：num = 3749

输出： "MMMDCCXLIX"

解释：

3000 = MMM 由于 1000 (M) + 1000 (M) + 1000 (M)
 700 = DCC 由于 500 (D) + 100 (C) + 100 (C)
  40 = XL 由于 50 (L) 减 10 (X)
   9 = IX 由于 10 (X) 减 1 (I)
注意：49 不是 50 (L) 减 1 (I) 因为转换是基于小数位
示例 2：

输入：num = 58

输出："LVIII"

解释：

50 = L
 8 = VIII
示例 3：

输入：num = 1994

输出："MCMXCIV"

解释：

1000 = M
 900 = CM
  90 = XC
   4 = IV
"""


# # 获取个位数
# ones = num % 10


# # 获取十位数
# tens = (num // 10) % 10

# # 获取百位数
# hundreds = (num // 100) % 10

# # 获取千位数
# thousands = (num // 1000) % 10


class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        # 解法1
        # 定义罗马数字符号及其对应的值
        # 求出个、十、百、千 位的数字
        n0, n1, n2, n3 = num % 10, (num % 100) // 10, (num % 1000) // 100, num // 1000

        # 对于 个、十、百 位分别求出对应的罗马数字
        def roman(n, s_lower, s_mid, s_higher):
            if n < 4: return s_lower * n
            if n == 4: return s_lower + s_mid
            if n < 9: return s_mid + s_lower * (n - 5)
            if n == 9: return s_lower + s_higher

        res = ''
        if n3: res += n3 * 'M'  # 数字范围<=3999 千位的转化是唯一的
        if n2: res += roman(n2, 'C', 'D', 'M')
        if n1: res += roman(n1, 'X', 'L', 'C')
        if n0: res += roman(n0, 'I', 'V', 'X')
        return res

        # 我的解法
        roman_str = ''
        k = (num // 1000) % 10
        roman_str = 'M' * k

        h = (num // 100) % 10
        if h == 5:
            roman_str = roman_str + 'D'
        if h < 5:
            if h == 4:
                roman_str = roman_str + 'C' + 'D'
            else:
                roman_str = roman_str + 'C' * h

        if h > 5:
            if h == 9:
                roman_str = roman_str + 'C' + 'M'
            else:
                roman_str = roman_str + 'D' + 'C' * (h - 5)

        s = (num // 10) % 10
        if s == 5:
            roman_str = roman_str + 'L'
        if s < 5:
            if s == 4:
                roman_str = roman_str + 'X' + 'L'
            else:
                roman_str = roman_str + 'X' * s
        if s > 5:
            if s == 9:
                roman_str = roman_str + 'X' + 'C'
            else:
                roman_str = roman_str + 'L' + 'X' * (s - 5)
        g = num % 10
        if g == 5:
            roman_str = roman_str + 'V'
        if g < 5:
            if g == 4:
                roman_str = roman_str + 'I' + 'V'
            else:
                roman_str = roman_str + 'I' * g
        if g > 5:
            if g == 9:
                roman_str = roman_str + 'I' + 'X'
            else:
                roman_str = roman_str + 'V' + 'I' * (g - 5)

        return roman_str
