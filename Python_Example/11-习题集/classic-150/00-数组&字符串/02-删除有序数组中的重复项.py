"""
给你一个 非严格递增排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，
返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。

考虑 nums 的唯一元素的数量为 k ，你需要做以下事情确保你的题解可以被通过：

更改数组 nums ，使 nums 的前 k 个元素包含唯一元素，并按照它们最初在 nums 中出现的顺序排列。nums 的其余元素与 nums 的大小不重要。
返回 k 。
"""


class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 使用双指针，快指针&慢指针
        # 快指针指向下一个元素
        # 慢指针指向上一个元素--指向可覆盖的位置

        # if len(nums) == 0:
        # 来检查数组是否为空
        if not nums:
            return 0
        # 慢指针
        # 初始化指针i，指向下一个可覆盖的位置,也就是新数组的最后一个位置，刚好也能体现不重复元素的个数
        i = 1
        # 第一个元素总是要保留的
        for j in range(1, len(nums)):
            # 不相等则覆盖并更新i,j指针，相等只更新j的指针
            if nums[j - 1] != nums[j]:
                nums[i] = nums[j]
                i += 1
        return i


nums = [1, 1, 1, 2, 2, 3, 3]
solution = Solution()
length = solution.removeDuplicates(nums)
print("New length:", length)  # 输出: 3
print("Modified array:", nums[:length])  # 输出: [1, 2, 3]
