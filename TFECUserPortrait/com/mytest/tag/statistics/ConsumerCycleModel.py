from pyspark.sql import SparkSession, DataFrame
import os

from cn.mytest.tag.base.AbstractBaseModel import AbstractBaseModel
import pyspark.sql.functions as F

"""
-------------------------------------------------
   Description :	TODO：统计类标签-消费周期
   SourceFile  :	ConsumerCycleModel
   Author      :	mytest team
-------------------------------------------------
"""

# 0.设置系统环境变量
os.environ['JAVA_HOME'] = '/export/server/jdk1.8.0_241/'
os.environ['SPARK_HOME'] = '/export/server/spark'
os.environ['PYSPARK_PYTHON'] = '/root/anaconda3/envs/pyspark_env/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/anaconda3/envs/pyspark_env/bin/python'


class ConsumerCycleModel(AbstractBaseModel):
    def compute(self, es_df: DataFrame, five_df: DataFrame):
        es_df.printSchema()
        # 默认显示前20条
        es_df.show()
        # 根据memberid分组，取每组中消费时间的最大值
        # F.current_date()：求当前的时间
        # F.from_unixtime("finishtime","yyyy-MM-dd")：把时间戳类型转换为yyyy-MM-dd的格式
        es_df = es_df.groupBy("memberid").agg(F.max("finishtime").alias("finishtime")) \
            .select("memberid", F.datediff(F.current_date(), F.from_unixtime("finishtime", "yyyy-MM-dd")).alias("days"))
        es_df.printSchema()
        es_df.show()
        # 对标签规则数据进行切割，转换为id，start，end三个字段
        five_df = five_df.select("id", F.split("rule", "-")[0].alias("start"), F.split("rule", "-")[1].alias("end"))
        # 把业务数据es_df和标签规则数据进行join操作，取业务数据的memberid和标签规则中的id
        new_df = es_df.join(other=five_df, on=es_df["days"].between(five_df['start'], five_df['end']), how='left') \
            .select(es_df['memberid'].alias("userId"), five_df['id'].alias("tagsId"))
        five_df.printSchema()
        five_df.show()
        new_df.printSchema()
        new_df.show()
        return new_df


if __name__ == '__main__':
    consumerCycleModel = ConsumerCycleModel(23)
    consumerCycleModel.execute()
