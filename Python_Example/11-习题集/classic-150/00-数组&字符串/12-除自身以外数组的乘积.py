"""
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请 不要使用除法，且在 O(n) 时间复杂度内完成此题。



示例 1:

输入: nums = [1,2,3,4]
输出: [24,12,8,6]
示例 2:

输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]

"""


class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]

        nums=[1,2,3,4]
        2对应的就是1*3*4
        其中1就是前缀乘积，3*4就是后缀乘积。

        所以，我们新建2个列表作为前缀和后缀列表(这里第一个都写成1)。
        pre=[1,1,2,6,24]
        suf=[1,4,12,24,24]
        那么只需要满足ans[i]=pre[i]*suf[len-i]即可。

        即ans[0]=pre[0]*suf[3]=1*24=24
        ans[1]=pre[1]*suf[2]=1*12=12
        ans[2]=pre[2]*suf[1]=2*4=8
        ans[3]=pre[3]*suf[0]=6*1=6

        具体如下：
        原数组：       [1       2       3       4]
        左部分的乘积：  [1]       1      1*2    1*2*3
        右部分的乘积： 2*3*4    3*4      4      [1]
        结果：        1*2*3*4  1*3*4   1*2*4  1*2*3*1

        其中前缀乘积和后缀乘积中的[1]为前缀的初始值1

        """
        n = len(nums)

        # 创建存放前缀乘积和后缀乘积的数组
        # [1]*3 = [1, 1, 1]
        # 初始化 left_products 和 right_products：
        # left_products[0] = 1，因为第一个元素左侧没有元素，所以乘积为1。
        # right_products[n-1] = 1，因为最后一个元素右侧没有元素，所以乘积为1。
        left_products = [1] * n
        right_products = [1] * n
        answer = [0] * n

        # 计算前缀乘积
        for i in range(1, n):
            left_products[i] = left_products[i - 1] * nums[i - 1]

        # 计算后缀乘积--表示从 n - 2 开始，到 -1 结束，但不包括 -1，每次递减 1
        # n = 5  # 数组的长度
        # for i in range(n - 2, -1, -1):
        #     print(i)
        # 3
        # 2
        # 1
        # 0

        # 计算右侧乘积
        for i in range(n - 2, -1, -1):
            right_products[i] = right_products[i + 1] * nums[i + 1]

        # 计算答案
        for i in range(n):
            answer[i] = left_products[i] * right_products[i]

        return answer


obj = Solution()
# 示例用法
nums = [1, 2, 3, 4]
print(obj.productExceptSelf(nums))  # 输出 [24, 12, 8, 6]
