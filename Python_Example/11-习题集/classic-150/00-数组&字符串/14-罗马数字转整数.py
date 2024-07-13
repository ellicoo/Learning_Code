"""
罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。



示例 1:

输入: s = "III"
输出: 3
示例 2:

输入: s = "IV"
输出: 4
示例 3:

输入: s = "IX"
输出: 9
示例 4:

输入: s = "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
示例 5:

输入: s = "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.


思路：
罗马数字由 I,V,X,L,C,D,M 构成；
当小值在大值的左边，则减小值，如 IV=5-1=4；
当小值在大值的右边，则加小值，如 VI=5+1=6；
由上可知，右值永远为正，因此最后一位必然为正。
一言蔽之，把一个小值放在大值的左边，就是做减法，否则为加法。
"""


class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        # luoma_nums = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        # int_nums = [1, 5, 10, 50, 100, 500, 1000]
        # luoma_nums_mapping_int_nums = dict(zip(luoma_nums,int_nums))

        roman_to_int_map = {
            'I': 1, 'V': 5, 'X': 10,
            'L': 50, 'C': 100, 'D': 500,
            'M': 1000
        }

        sum = 0
        n = len(s)
        # 字符串也是容器类型，可以使用for遍历
        for i in range(n):
            if i < n - 1 and roman_to_int_map[s[i]] < roman_to_int_map[s[i + 1]]:
                sum -= roman_to_int_map[s[i]]
            else:
                sum += roman_to_int_map[s[i]]

        return sum




