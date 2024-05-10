from pyspark import SparkConf, SparkContext
from my_utils.get_local_file_system_absolute_path import get_absolute_path
import os
# 配置SPARK_HOME的路径
os.environ['SPARK_HOME'] = '/export/server/spark'
# 配置base环境Python解析器的路径
os.environ['PYSPARK_PYTHON'] = '/root/anaconda3/bin/python3'
# 配置base环境Python解析器的路径
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/anaconda3/bin/python3'

#1.构建Spark环境
conf = SparkConf().setAppName("wordcount").setMaster("local")
sc = SparkContext(conf=conf)

#2.读取数据
input_rdd = sc.textFile(get_absolute_path("../data/word_re.txt"))

#3.处理数据
result_rdd = input_rdd.flatMap(lambda line : line.split(" "))\
    .map(lambda word : (word,1))\
    .reduceByKey(lambda x,y : x + y)

#4.输出数据
result_rdd.foreach(lambda x : print(x))

#5.停止Spark环境
sc.stop()
