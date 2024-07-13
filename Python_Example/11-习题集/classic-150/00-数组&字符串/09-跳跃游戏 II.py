"""
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i]
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。



示例 1:

输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
示例 2:

输入: nums = [2,3,0,1,4]
输出: 2

思路：使用贪心算法
"""


class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n <= 1:
            return 0

        jumps = 0  # 记录跳跃次数
        current_end = 0  # 记录当前索引发生更改的时机
        farthest = 0  # 当前能到达的最远位置，总是找到最远的的距离

        for i in range(n - 1):
            # 遍历索引，更新当前是否能够走得更远
            farthest = max(farthest, i + nums[i])

            # 如果当前索引到达了 current_end，增加 jumps 变量，并更新 current_end 为 farthest。
            if i == current_end:
                jumps += 1
                # 如果当前索引到达了 current_end，增加 jumps 变量，并更新 current_end 为 farthest。
                current_end = farthest
                if current_end >= n - 1:
                    break

        return jumps


nums1 = [2, 3, 1, 1, 4]
nums2 = [3, 2, 1, 0, 4]
obj = Solution()
print(obj.jump(nums1))
print(obj.jump(nums2))
