# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 辅助函数：将列表转换为链表
def list_to_linkedlist(lst):
    if not lst:
        return None
    head = ListNode(lst[0])
    current = head
    for value in lst[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# 辅助函数：将链表转换为列表
def linkedlist_to_list(node):
    lst = []
    while node:
        lst.append(node.val)
        node = node.next
    return lst


class Solution(object):
    def mergeTwoLists(self, list1, list2):

        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]

        """
        # 创建一个哑节点
        dummy = ListNode()
        p = dummy  # p用于往后连新节点
        while list1 and list2:
            if list1.val <= list2.val:  # p连接l1或者l2中更小的节点
                p.next = list1
                list1 = list1.next
            else:
                p.next = list2
                list2 = list2.next
            p = p.next  # p指新链表的尾节点
        # p连上l1或l2中剩下的那一个节点
        p.next = list1 if list1 is not None else list2
        return dummy.next

# 测试用例
obj = Solution()
l1 = list_to_linkedlist([1, 2, 4])
l2 = list_to_linkedlist([1, 3, 4])
merged_list = obj.mergeTwoLists(l1, l2)
print(linkedlist_to_list(merged_list))  # 输出: [1, 1, 2, 3, 4, 4]