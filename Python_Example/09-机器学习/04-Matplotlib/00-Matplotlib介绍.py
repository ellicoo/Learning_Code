"""
Matplotlib 是一个用于绘制图形和可视化数据的Python库。它提供了丰富的绘图功能，
可以用于创建各种静态、动态和交互式图表，适用于数据分析、科学计算、工程学和其他领域。
"""
#以下是 Matplotlib 的主要特点和用法：

#（1）简单易用的接口： Matplotlib 提供了简单直观的绘图接口，使用户能够轻松创建各种类型的图表，包括线图、散点图、柱状图、饼图等。
import matplotlib.pyplot as plt

# 创建一个简单的线图
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('简单线图')
plt.show()

#（2）多种图形类型： Matplotlib 支持多种图形类型和样式，使用户能够自定义图表的外观和风格。可以通过参数调整颜色、线型、标记等
# 创建散点图
plt.scatter(x, y, color='red', marker='o')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('散点图')
plt.show()

#（3）子图和布局： Matplotlib 允许用户创建多个子图，实现更复杂的布局。这对于同时比较多个数据集或绘制多个相关的图表非常有用。
# 创建多个子图
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title('子图 1')

plt.subplot(2, 1, 2)
plt.scatter(x, y, color='green', marker='x')
plt.title('子图 2')

plt.show()

#（4）3D图形： Matplotlib 支持绘制3D图形，包括三维散点图、曲面图等，适用于展示具有三维关系的数据。
from mpl_toolkits.mplot3d import Axes3D

# 创建3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, [1, 2, 3, 4, 5], c='blue', marker='o')

ax.set_xlabel('X轴标签')
ax.set_ylabel('Y轴标签')
ax.set_zlabel('Z轴标签')

plt.show()

#（5）可交互性和动画： Matplotlib 提供了一些工具和API，使用户能够创建交互式图形和动画，以更好地探索和展示数据。
# 创建动画
# from matplotlib.animation import FuncAnimation
#
# def update(frame):
#     # 更新图形的函数
#     # ...
#     pass
#
# ani = FuncAnimation(fig, update, frames=range(10), blit=True)
# plt.show()
