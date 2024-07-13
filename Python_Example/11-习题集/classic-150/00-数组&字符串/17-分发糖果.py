"""
n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻两个孩子评分更高的孩子会获得更多的糖果。
请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。



示例 1：

输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
示例 2：

输入：ratings = [1,2,2]
输出：4
解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
"""

# 思路：贪心算法--

class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """

        n = len(ratings)
        if n == 0:
            return 0

        # 初始化每个孩子至少有一颗糖果
        candies = [1] * n

        # 从左到右遍历，确保右边比左边高一颗糖
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        # 从右到左遍历，确保左边比右边高一颗糖
        for i in range(n - 1, 0, -1):
            if ratings[i - 1] > ratings[i]:
                # 如果相邻前一个数更大，那前一个孩子就要比后一个孩子多一个糖果
                candies[i - 1] = max(candies[i - 1], candies[i] + 1)

        # 返回糖果总数
        return sum(candies)

        # 1，5，2，3，1
        # 1，2，1，2，1







