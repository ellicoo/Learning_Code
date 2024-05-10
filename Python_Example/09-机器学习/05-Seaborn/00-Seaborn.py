"""
Seaborn 是一个基于 Matplotlib 的数据可视化库，专注于绘制统计图形。
它提供了高层次的接口，使得绘制各种统计图形变得简单而直观。Seaborn
的设计目标是创建具有良好美观度的图形，并且在可视化统计数据时更加方便。
"""
# 以下是 Seaborn 的一些主要特点和功能：
# （1）高层次的接口： Seaborn 提供了简单而灵活的API，可以轻松创建多种统计图形，包括散点图、线图、箱线图、直方图、热力图等。
import seaborn as sns
import matplotlib.pyplot as plt

# 创建散点图
from pandas.conftest import iris

sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species')

# 显示图形
plt.show()

#（2）内置主题和颜色： Seaborn 提供了多种内置主题和颜色调色板，使得图形更加美观和专业。通过设置主题，可以调整整个图形的外观。
# 设置主题为白色背景
sns.set_theme(style="whitegrid")

# 创建箱线图
sns.boxplot(x='species', y='sepal_length', data=iris)

# 显示图形
plt.show()

#（3）统计估计和自动注释： Seaborn 具有内置的统计估计功能，例如绘制散点图时，可以自动计算并绘制回归线。此外，Seaborn 还支持在图形中自动注释数据点的功能。
# 创建散点图并添加回归线
sns.regplot(x='sepal_length', y='sepal_width', data=iris)

# 显示图形
plt.show()

#（4）分组图形： Seaborn 支持通过 hue 参数实现简单的分组图形，例如通过不同颜色或样式区分数据的不同类别。
# 创建分组的直方图
sns.histplot(data=iris, x='sepal_length', hue='species', multiple='stack')

# 显示图形
plt.show()

#（5）适应数据框： Seaborn 对 Pandas 数据框（DataFrame）友好，可以直接使用数据框的列作为绘图的输入，减少了数据准备的工作。
# 使用数据框的列创建箱线图
sns.boxplot(x='species', y='sepal_length', data=iris)

# 显示图形
plt.show()
