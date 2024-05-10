# coding:utf8
import os

import jieba

"""
-------------------------------------------------
   Description :	TODO：action算子开始
   SourceFile  :	Demo05_MapFunction
   Author      :	81196
   Date	       :	2023/9/7
-------------------------------------------------
"""

# 0.设置系统环境变量
os.environ['JAVA_HOME'] = '/export/server/jdk1.8.0_241/'
os.environ['HADOOP_HOME'] = '/export/server/hadoop'
os.environ['PYSPARK_PYTHON'] = '/root/anaconda3/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/anaconda3/bin/python3'


def context_jieba(data):
    """通过jieba分词工具 进行分词操作"""
    seg = jieba.cut_for_search(data)
    l = list()
    for word in seg:
        l.append(word)
    return l


def filter_words(data):
    """过滤不要的 谷 \ 帮 \ 客"""
    return data not in ['谷', '帮', '客']


def append_words(data):
    """修订某些关键词的内容"""
    if data == '传智播': data = '传智播客'
    if data == '院校': data = '院校帮'
    if data == '博学': data = '博学谷'
    return (data, 1) # 为节省一个rdd的map迭代，直接返回键值对


def extract_user_and_word(data):
    """传入数据是 元组 (1, 我喜欢传智播客)"""
    user_id = data[0]
    content = data[1]
    # 对content进行分词
    words = context_jieba(content)

    return_list = list()
    for word in words:
        # 不要忘记过滤 \谷 \ 帮 \ 客
        if filter_words(word):
            return_list.append((user_id + "_" + append_words(word)[0], 1))

    return return_list
