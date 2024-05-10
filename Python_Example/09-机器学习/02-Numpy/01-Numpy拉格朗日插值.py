# 可以使用 NumPy 库中的 numpy.polyfit 函数来进行拉格朗日插值。以下是一个简单的例子：
import numpy as np
import matplotlib.pyplot as plt

# 数据点
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 0, 1, 3, 4])

# 使用 numpy.polyfit 进行拉格朗日插值
coefficients = np.polyfit(x, y, len(x) - 1)

# 构建插值多项式
poly_interpolation = np.poly1d(coefficients)

# 生成插值结果
x_interpolation = np.linspace(min(x), max(x), 100)
y_interpolation = poly_interpolation(x_interpolation)

# 绘制原始数据和插值结果
plt.scatter(x, y, label='原始数据')
plt.plot(x_interpolation, y_interpolation, label='拉格朗日插值', linestyle='--', color='red')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()
plt.show()
