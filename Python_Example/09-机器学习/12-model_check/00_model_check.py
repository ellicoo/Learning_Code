from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import expr
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DecimalType

# TP：human为1，predi_human预测也为1，预测对了，预测正确--原本1，预测1
# TN：human为0，predi_human预测也为0，预测对了，预测正确--原本0，预测0
# FP：human为0，predi_human预测为1，错误的预测为正例，预测错了--原本非1，预测1
# FN：human为1，predi_human预测为0，错误的预测为反例，预测错了--原本非0，预测0

# Precision精确率: 表示在所有预测分类为正例中，有多少是真正的正例。---预测对的/所有预测为正的(对预测结果而言)
# Recall 召回率: 所有正样本中，所有被预测为正的有多少。--预测对的/所有正样本中(对样本而言)
# Accuracy 准确率: （本需求暂时不需要，先忽略）
# F1 score :F1分数是Precision精确率和Recall召回率的调和平均值，用于综合考虑分类器的性能,特别是在类别不平衡的情况下

# 精确率和召回率的问题：如果样本中大多数都是human=1的正样本，阀值将不具有辨识能力。阀值会将predi_human都倾向于输出为1
# 此时精确率和召回率的值都非常高。此时这个阀值就比较糟糕。所有当样本出现不均衡的情况下，我们将使用ROC曲线(AUC)进行评判：
# 使用两个指标计算：TPR=tp/(tp+fn)
#                FPR=fp/(fp+tn)
# 坐标点（TPR,FPR）


# F1属于[0~1]，其中1表示完美的分类器，0表示最差的分类器。F1分数在处理不平衡类别的情况下特别有用，因为它考虑了假正例和假负例，而不仅仅是准确性。在某些情况下，精确率和召回率之间存在权衡，F1分数帮助找到了一个平衡点，使得分类器在精确率和召回率之间取得良好的性能。

risk_human_table = spark.read.table("airbot_prod.bot_master_all_risk_level_score")

# 表名temp_risk_human_table
risk_table = risk_human_table.select("score", "human")
# 添加新列 predi_human
threshold = 10
# predi_human_table = risk_table.withColumn("predi_human", when(col("score") >= threshold, 0).otherwise(1).cast(DecimalType(10, 0)))
predi_human_table = risk_table.withColumn("predi_human", when(col("score") >= threshold, 0).otherwise(1))
# 显示结果
# predi_human_table.show()


# 混淆矩阵
result_df = predi_human_table.select("human","predi_human")
conf_matrix = result_df.groupBy("human", "predi_human").count()
conf_matrix.show()

# 将 DecimalType(10, 0) 转换为 DoubleType
result_df = result_df.withColumn("predi_human", col("predi_human").cast(DoubleType()))

# 计算 FP、FN、TP、TN
fp = result_df.filter((col("human") == 0) & (col("predi_human") == 1.0)).count()
fn = result_df.filter((col("human") == 1) & (col("predi_human") == 0.0)).count()
tp = result_df.filter((col("human") == 1) & (col("predi_human") == 1.0)).count()
tn = result_df.filter((col("human") == 0) & (col("predi_human") == 0.0)).count()

print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")

# 计算 ROC 和 AUC
# 这个代码算出的是 AUC（Area Under the ROC Curve），而不是 ROC 曲线的具体坐标点。BinaryClassificationEvaluator 中 metricName # 参数设置为 "areaUnderROC" 时，它会计算 ROC 曲线下的面积，即 AUC。所以，roc_auc 是 AUC 的值，而不是 ROC 曲线的详细坐标点。
evaluator = BinaryClassificationEvaluator(labelCol="human", rawPredictionCol="predi_human", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(result_df)
print(f"ROC AUC: {roc_auc}")

# 计算 Precision、Recall、Accuracy、F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# accuracy = (tp + tn) / (tp + tn + fp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
# print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")