# 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
# 请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
# 注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，
# 其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。

def merge_num1_num2tonum1(nums1, m, nums2, n):
    # 第一步：先初始化两个索引指针，分别指向 nums1 和 nums2 的末尾
    p1, p2 = m - 1, n - 1

    # 第二步：再初始化合并后的数组指针，指向 nums1 的末尾
    p = m + n - 1

    # 第三步：再从后往前遍历，将较大的元素放入 nums1 的末尾
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1


# 测试数据集合--list列表
nums1 = [1, 2, 3, 0, 0, 0, 0]
m = 3
nums2 = [2, 5, 6, 4]
n = 4
merge_num1_num2tonum1(nums1, m, nums2, n)
print(nums1)


def merge_num1_num2_toresult(nums1, m, nums2, n):
    # 初始化两个指针，分别指向 nums1 和 nums2 的开头
    p1, p2 = 0, 0

    # 初始化结果数组
    result = []

    # 依次比较两个数组的元素，将较小的元素放入结果数组
    while p1 < m and p2 < n:
        if nums1[p1] < nums2[p2]:
            result.append(nums1[p1])
            p1 += 1
        else:
            result.append(nums2[p2])
            p2 += 1

    # 将剩余的元素复制到结果数组中
    while p1 < m:
        result.append(nums1[p1])
        p1 += 1

    while p2 < n:
        result.append(nums2[p2])
        p2 += 1

    return result


# 示例
nums1 = [1, 2, 3, 4]
m = 4
nums2 = [2, 5, 6]
n = 3

result_array = merge_num1_num2_toresult(nums1, m, nums2, n)
print(result_array)
