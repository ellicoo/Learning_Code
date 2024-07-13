"""
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释:
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
"""


# 方法1：
# 时间&空间复杂度都是O(n)
class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n  # Handle the case where k is greater than the length of the array
        nums[:] = nums[-k:] + nums[:-k]


# 方法2：时间复杂度O(n)空间复杂度O(1)
# class Solution:
#     def rotate(self, nums, k):
#         """
#         :type nums: List[int]
#         :type k: int
#         :rtype: None Do not return anything, modify nums in-place instead.
#         """
#         n = len(nums)
#         k = k % n
#
#         self.reverse(nums, 0, n - 1)
#         self.reverse(nums, 0, k - 1)
#         self.reverse(nums, k, n - 1)
#
#     def reverse(self, nums, start, end):
#         while start < end:
#             nums[start], nums[end] = nums[end], nums[start]
#             start += 1
#             end -= 1


nums = [1, 2, 3, 4, 5, 6, 7]
print(nums)
k = 3
obj = Solution()
obj.rotate(nums, k)
print(nums)
