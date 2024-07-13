"""
给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
"""

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 2:
            return len(nums)

        # 初始化指针i，指向下一个可覆盖的位置
        # 使用两个指针来遍历数组，这个指针指向修改后数组的下一个位置
        i = 2 # 指针指向新数组的第三个位置
        # 指针初始化：从第三个元素位置开始，j指针指向数组当前遍历到的位置，
        for j in range(2, len(nums)):
            # 我需要把不重复的元素给存进新数组
            if nums[j] != nums[i - 2]:
                nums[i] = nums[j]
                i += 1
        return i


nums = [1, 1, 1, 2, 2, 3]
solution = Solution()
length = solution.removeDuplicates(nums)
print("New length:", length)  # 输出: 5
print("Modified array:", nums[:length])  # 输出: [1, 1, 2, 2, 3]
