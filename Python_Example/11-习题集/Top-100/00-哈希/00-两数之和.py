class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 键存的是数组的值
        # 值存的是数组的索引
        num_map = {}
        """
        使用 enumerate 函数遍历数组 nums，同时获取元素 num 和其下标 i。
        计算当前元素与目标值的差 complement = target - num。
        在哈希表 num_map 中查找 complement：
        如果找到了 complement，说明存在两个元素的和等于 target，返回它们的下标 [num_map[complement], i]。
        如果没找到，将当前元素 num 及其下标 i 存入 num_map 中 num_map[num] = i。
        时间复杂度O(n)，hash操作O(1)
        """
        for i, num in enumerate(nums):
            complement = target - num
            # # 检查 num 是否是字典 num_map 的一个键
            if complement in num_map:
                # 找到就返回
                return [num_map[complement], i]
            # 不在，就存储值和索引，因为有很多组合，所以需要存储
            num_map[num] = i
        return []


# 示例用法
nums = [2, 7, 11, 15]
target = 9
solution = Solution()
print(solution.twoSum(nums, target))  # 输出: [0, 1]
