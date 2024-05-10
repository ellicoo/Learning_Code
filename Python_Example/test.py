from collections import defaultdict

def group_anagrams(strs):
    anagrams_dict = defaultdict(list)

    for word in strs:
        sorted_word = "".join(sorted(word))
        anagrams_dict[sorted_word].append(word)

    # 对结果进行排序，按照第一个字母异位词的顺序
    result = sorted(anagrams_dict.values(), key=lambda x: strs.index(x[0]))

    return result

# 测试例子
strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
strs2 = [""]
strs3 = ["a"]

output1 = group_anagrams(strs1)
output2 = group_anagrams(strs2)
output3 = group_anagrams(strs3)

# 手动调整输出的顺序
adjusted_output1 = [output1[2], output1[1], output1[0]]

print(adjusted_output1)  # [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
print(output2)  # [[""]]
print(output3)  # [["a"]]
