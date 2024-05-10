from pyspark.sql import SparkSession, DataFrame
import os

from pyspark.sql.types import StringType

from cn.mytest.tag.bean.EsMeta import EsMeta, tagRuleStrToEsMeta
import pyspark.sql.functions as F

"""
-------------------------------------------------
   Description :	TODO：基类（父类）
   SourceFile  :	AbstractBaseModel
   Author      :	mytest team
-------------------------------------------------
"""

# 0.设置系统环境变量
os.environ['JAVA_HOME'] = '/export/server/jdk1.8.0_241/'
os.environ['SPARK_HOME'] = '/export/server/spark'
os.environ['PYSPARK_PYTHON'] = '/root/anaconda3/envs/pyspark_env/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/anaconda3/envs/pyspark_env/bin/python'


# 自定义的函数，用来实现标签的合并
@F.udf(returnType=StringType())
def merge_tags(new_df, old_df,fiveTagIDStr):
    # 1.如果new_df为空，返回old_df数据
    if new_df is None:
        return old_df
    # 2.如果old_df为空，返回new_df数据
    if old_df is None:
        return new_df

    # 3.如果两个都不为空，实现标签的合并
    # 3.1 new_df切割，得到一个列表，new_df_list
    new_df_list = str(new_df).split(",")
    # 3.2 old_df切割，得到一个列表，old_df_list
    old_df_list = str(old_df).split(",")

    #fiveTagIDStr字符串中，包含了所有5级标签的ID，使用(,)拼接，因此需要使用(,)切割
    five_tag_id_list = fiveTagIDStr.split(",")
    for tag in five_tag_id_list:
        if tag in old_df_list:
            old_df_list.remove(tag)

    # 3.3 把new_df_list和old_df_list进行合并，得到result_df_list
    result_df_list = new_df_list + old_df_list
    # 3.4 把最终的result_df_list以固定的符号拼接，返回
    return ",".join(set(result_df_list))



class AbstractBaseModel:
    # 0.初始化Spark环境，一样的
    def __init__(self,fourTagId):
        # 构建SparkSession
        # 建造者模式：类名.builder.配置…….getOrCreate()
        # 自动帮你构建一个SparkSession对象，只要指定你需要哪些配置就可
        self.fourTagId = fourTagId
        self.spark = SparkSession \
            .builder \
            .master("local[2]") \
            .appName("SparkSQLAppName") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
            .config("spark.sql.shuffle.partitions", 4) \
            .getOrCreate()

    # 1.根据4级标签ID，读取MySQL标签体系数据，一样的
    def read_mysql_data(self):
        input_df = self.spark.read.jdbc(url='jdbc:mysql://up01:3306/tfec_tags',
                                   table='tbl_basic_tag',
                                   properties={"user": "root", "password": "123456"})
        return input_df

    # 2.过滤4级标签的数据，将四级标签的rule转换为esMeta对象，id不同，逻辑相同
    def input_df_to_esMeta(self,input_df):
        ruleStr = input_df.where(f"id = {self.fourTagId}").select("rule").first()[0]
        esMeta = EsMeta(**tagRuleStrToEsMeta(ruleStr))
        return esMeta

    # 3.根据esMeta对象从ES中读取相应的业务数据，一样的
    def read_es_df_by_esMeta(self,esMeta):
        es_df = self.spark.read \
            .format('es') \
            .option("es.resource", esMeta.esIndex) \
            .option("es.nodes", esMeta.esNodes) \
            .option("es.read.field.include", esMeta.selectFields) \
            .option("es.mapping.date.rich", "false") \
            .load()
        return es_df

    # 4.根据4级标签ID，读取5级标签的数据，一样的
    def read_five_df_by_fourTagId(self,input_df):
        five_df: DataFrame = input_df.where(f"pid = {self.fourTagId}").select("id", "rule")
        return five_df

    # 5.通过ES中的业务数据与MySQL的5级标签进行打标签，完全不一样，返回new_df
    def compute(self,es_df,five_df):
        pass

    # 6.从ES中读取历史用户标签数据，一样的
    def read_old_df_from_es(self,esMeta):
        old_df = self.spark.read \
            .format('es') \
            .option("es.resource", "tags_result") \
            .option("es.nodes", esMeta.esNodes) \
            .option("es.read.field.include", "userId,tagsId") \
            .option("es.mapping.date.rich", "false") \
            .load()
        return old_df

    # 7.将老的用户画像标签与新的标签进行合并，得到最终标签，一样的
    def merge_old_df_and_new_df(self,new_df,old_df,fiveTagIDStr):
        result_df = new_df.join(other=old_df, on=new_df['userId'] == old_df['userId'], how='left') \
            .select(new_df['userId'], merge_tags(new_df['tagsId'], old_df['tagsId'],fiveTagIDStr).alias("tagsId"))
        return result_df

    # 8.将最终的结果写入ES中，一样的
    def write_result_df_to_es(self,result_df,esMeta):
        result_df.write \
            .format("es") \
            .option("es.resource", "tags_result") \
            .option("es.nodes", esMeta.esNodes) \
            .option("es.mapping.id", "userId") \
            .mode("append") \
            .save()

    # 9.销毁Spark环境，释放资源，一样的
    def close(self):
        self.spark.stop()

    """
    自定义的execute方法，用于把上述的方法串起来执行
    """
    def execute(self):
        # # 0.初始化Spark环境
        self.__init__(self.fourTagId)
        # 1.根据4级标签ID，读取MySQL标签体系数据--读出来的结果是一张表
        input_df = self.read_mysql_data()
        # 2.过滤4级标签的数据，将四级标签的rule转换为esMeta对象，id不同，逻辑相同
        esMeta = self.input_df_to_esMeta(input_df)
        # 3.根据esMeta对象从ES中读取相应的业务数据--读出来的结果是一张表--一张待匹配或者待统计或者待挖掘的业务数据的表
        es_df = self.read_es_df_by_esMeta(esMeta)
        # 4.根据4级标签ID，读取5级标签的数据
        # 刚刚我们只要input(来源与mysql)表中的rule字段来转换成meta对象去es中找待操作的数据表
        # 现在我们还需要input表中的其他字段值，level子段为5的值，这个值有很多，但我只要刚刚指定的rule字段下的5级值，取出的是规则值
        # 把规则值和刚刚找到的es表的数据进行操作(匹配、统计、挖掘)
        five_df = self.read_five_df_by_fourTagId(input_df)
        # 5.通过ES中的业务数据与MySQL的5级标签进行打标签，完全不一样
        new_df = self.compute(es_df,five_df)
        try:
            # 6.从ES中读取历史用户标签数据
            old_df = self.read_old_df_from_es(esMeta)
            #标签更新
            """
            需要传入改4级标签下的所有5级标签的ID号。不能直接给five_df（ID、rule）
            （1）通过five_df获取所有ID
            （2）传入所有ID（List）List不能直接传入到自定义函数中，自定义需要只能传入2种类型（Column、Str）
            （3）把list转换为字符串，使用固定的分割符号拼接
                ",".join(List)
            """
            fiveTagIDList = five_df.rdd.map(lambda x:x.id).collect()

            # 7.将老的用户画像标签与新的标签进行合并，得到最终标签
            result_df = self.merge_old_df_and_new_df(new_df,old_df,F.lit(",".join(str(id)for id in fiveTagIDList)))
        except:
            #如果出了问题，说明这个任务是第一次运行，没有历史标签库
            #可以使用new_df当做标签结果数据，避免合并出问题
            result_df = new_df
            print("--------------首次执行，跳过合并---------------")
        # 8.将最终的结果写入ES中
        self.write_result_df_to_es(result_df,esMeta)
        # 9.销毁Spark环境，释放资源
        self.close()