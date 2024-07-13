"""
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。
"""


class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方1：
        # 字典：
        # 键存的是数组的值
        # 值存的是数组的索引
        # 最坏请看：时间&空间O（n）
        count = {}
        # 检查 num 是否是字典 count 的一个键
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
            # 向下取整
            if count[num] > len(nums) // 2:
                return num

        # 方法2：使用 Boyer-Moore 投票算法 时间O(n)空间 O(1)
        # count = 0
        # candidate = None
        #
        # for num in nums:
        #     if count == 0:
        #         candidate = num
        #     count += (1 if num == candidate else -1)
        #
        # return candidate


nums = [3, 2, 3]
obj = Solution()
v = obj.majorityElement(nums)
print(v)
