def quicksort(arr):
    # 基本情况: 如果列表长度小于等于1, 则返回列表
    if len(arr) <= 1:
        return arr
    else:
        # 选择基准点，这里选择列表的第一个元素
        pivot = arr[0]
        # 分割列表，分成小于基准点、等于基准点和大于基准点的三个子列表
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        # 递归调用quicksort并合并结果
        return quicksort(less_than_pivot) + [pivot] + quicksort(greater_than_pivot)


# 测试用例
arr = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(arr))  # 输出: [1, 1, 2, 3, 6, 8, 10]
