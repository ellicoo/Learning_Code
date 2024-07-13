"""
给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。

根据维基百科上 h 指数的定义：h 代表“高引用次数” ，一名科研人员的 h 指数 是指他（她）至少发表了 h 篇论文，并且 至少 有 h 篇论文被引用次数大于等于 h 。如果 h 有多种可能的值，h 指数 是其中最大的那个。



示例 1：

输入：citations = [3,0,6,1,5]
输出：3
解释：给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 3, 0, 6, 1, 5 次。
     由于研究者有 3 篇论文每篇 至少 被引用了 3 次，其余两篇论文每篇被引用 不多于 3 次，所以她的 h 指数是 3。
示例 2：

输入：citations = [1,3,1]
输出：1


示例1：
1  3 >= 1                                                                         max_count=1
2  3 >= 2  or  3>=1                                                               max_count=1
3  3 >= 3, 6 >= 3 不成立 or 3>=2,6>=2  or 3>=1 or 6>=1                             max_count=2
4  6 >= 4 or 6 >= 3,3>=3 or 6>=2,3>=2 or 6>=1,3>=1                                max_count=2
5  5 >=5,6>=5 不成立 or 5>=4,6>=4 不成立 or 5>=3,6>=3,3>=3 or 5>=2,6>=2,3>=2不成立   max_count=3


示例2：
1 1>=1                          max_count = 1
2 3>=2 or 3>=1,1>=1 不成立       max_count = 1
3 3>=3 or 1>=1,3>=1,1>=1 不成立  max_count = 1


思路：
最好的方法：有一种线性时间复杂度的算法可以用来计算 h 指数 ： 利用计数排序的思想，
通过创建一个计数数组来统计每个引用次数出现的频率，然后从后向前遍历计数数组来确定 h 指数1，将引用次数排序。

易懂的方法：
1。将引用次数排序。
2。从高到低遍历引用次数，找到最大的 h，使得至少有 h 篇论文的引用次数大于等于 h。


"""


class Solution(object):
    def h_index(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        # 方法1：易懂
        #
        # # 将引用次数排序
        citations.sort(reverse=True)

        h = 0
        # 遍历排序后的引用次数
        for i in range(len(citations)):
            if citations[i] >= i + 1:
                h = i + 1
            else:
                break

        return h

        # n = len(citations)
        # count = [0] * (n + 1)
        #
        # # 统计每个引用次数的出现频率
        # for c in citations:
        #     if c >= n:
        #         count[n] += 1
        #     else:
        #         count[c] += 1
        #
        # total = 0
        # # 从后向前遍历计数数组
        # for i in range(n, -1, -1):
        #     total += count[i]
        #     if total >= i:
        #         return i
        #
        # return 0


obj = Solution()

# 示例用法
citations1 = [3, 0, 6, 1, 5]
print(obj.h_index(citations1))  # 输出: 3

citations2 = [1, 3, 1]
print(obj.h_index(citations2))  # 输出: 1

citations3 = [10, 8, 5, 4, 3]
print(obj.h_index(citations3))  # 输出: 4
