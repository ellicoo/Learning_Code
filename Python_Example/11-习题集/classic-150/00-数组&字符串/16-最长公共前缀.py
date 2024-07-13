"""
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。



示例 1：

输入：strs = ["flower","flow","flight"]
输出："fl"
示例 2：

输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
"""


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # zip *就是把字符串的字母挨个各个取出一个，组成一个新元组,暂存在temp中，temp其实是一个元组
        # 转成集合去存，如果是公共前缀，那么遍历出来的每个字符都相同，则去重后的长度就是1
        re = ""
        for temp in zip(*strs):
            temp_char_set = set(temp)
            if len(temp_char_set) == 1:
                re += temp[0]
            else:
                break
        return re


strs = ["dog", "racecar", "car"]
obj = Solution()
longs_count = obj.longestCommonPrefix(strs)
print(longs_count)
