"""
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。



示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
示例 2：

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

# 思路： 使用贪心算法来解决这个问题。贪心算法的核心思想是尽可能跳到最远的位置，同时检查是否能够达到最后一个位置。

"""


class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        max_reach = 0  # 记录当前能够达到的最远位置

        for i in range(len(nums)):
            if i > max_reach:
                return False  # 如果当前位置超过了最远能到达的位置，则无法继续前进

            max_reach = max(max_reach, i + nums[i])  # 更新最远能到达的位置

            if max_reach >= len(nums) - 1:
                return True  # 如果最远能到达的位置超过或者等于最后一个位置，返回True

        return False


nums1 = [2, 3, 1, 1, 4]
nums2 = [3, 2, 1, 0, 4]
obj = Solution()
print(obj.canJump(nums1))
print(obj.canJump(nums2))
