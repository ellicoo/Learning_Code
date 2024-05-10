def groupAnagrams(strs):
    anagrams = {}

    for word in strs:
        sorted_word = ''.join(sorted(word))
        if sorted_word in anagrams:
            anagrams[sorted_word].append(word)
        else:
            anagrams[sorted_word] = [word]

    # 按照要求对字典的值进行排序
    result = sorted(anagrams.values(), key=lambda x: (len(x), x[0]))
    return result


# 测试案例
strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(groupAnagrams(strs1))  # 输出: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]

strs2 = [""]
print(groupAnagrams(strs2))  # 输出: [[""]]

strs3 = ["a"]
print(groupAnagrams(strs3))  # 输出: [["a"]]