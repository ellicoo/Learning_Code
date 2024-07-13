"""
实现RandomizedSet 类：

RandomizedSet() 初始化 RandomizedSet 对象
bool insert(int val) 当元素 val 不存在时，向集合中插入该项，并返回 true ；否则，返回 false 。
bool remove(int val) 当元素 val 存在时，从集合中移除该项，并返回 true ；否则，返回 false 。
int getRandom() 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有 相同的概率 被返回。
你必须实现类的所有函数，并满足每个函数的 平均 时间复杂度为 O(1) 。



示例：

输入
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
输出
[null, true, false, true, 2, true, false, 2]

解释
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // 向集合中插入 1 。返回 true 表示 1 被成功地插入。
randomizedSet.remove(2); // 返回 false ，表示集合中不存在 2 。
randomizedSet.insert(2); // 向集合中插入 2 。返回 true 。集合现在包含 [1,2] 。
randomizedSet.getRandom(); // getRandom 应随机返回 1 或 2 。
randomizedSet.remove(1); // 从集合中移除 1 ，返回 true 。集合现在包含 [2] 。
randomizedSet.insert(2); // 2 已在集合中，所以返回 false 。
randomizedSet.getRandom(); // 由于 2 是集合中唯一的数字，getRandom 总是返回 2 。

"""


import random

class RandomizedSet:
    def __init__(self):
        self.val_to_index = {}
        self.values = []

    def insert(self, val: int) -> bool:

        if val in self.val_to_index:
            return False
        # 如果不存在，将元素插入到列表的末尾，并在哈希表中记录其索引。
        self.val_to_index[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_index:
            return False
        # 获取要删除的值的索引
        index = self.val_to_index[val]
        # 将列表的最后一个元素移动到要删除元素的位置
        last_val = self.values[-1]
        self.values[index] = last_val
        self.val_to_index[last_val] = index
        # 删除最后一个元素
        self.values.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.values)

# 示例用法
randomizedSet = RandomizedSet()
print(randomizedSet.insert(1))  # 返回 True
print(randomizedSet.remove(2))  # 返回 False
print(randomizedSet.insert(2))  # 返回 True
print(randomizedSet.getRandom())  # 随机返回 1 或 2
print(randomizedSet.remove(1))  # 返回 True
print(randomizedSet.insert(2))  # 返回 False
print(randomizedSet.getRandom())  # 总是返回 2

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
