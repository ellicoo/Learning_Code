"""
Scipy（Scientific Python）是一个用于科学计算的开源库，建立在NumPy之上，
提供了许多高级的数学、科学和工程计算功能。Scipy包含了一系列的子模块，
每个子模块都专注于不同领域的科学计算任务，如优化、信号处理、统计学、插值等。

NumPy 和 SciPy 是两个在科学计算和数据分析中广泛使用的 Python 库，
它们提供了不同的功能，但通常是协同工作的。以下是它们的主要区别和使用场景：

NumPy：

功能： NumPy 主要提供了多维数组对象（numpy.ndarray）以及对数组进行操作的函数。
它是 Python 中进行科学计算的基础库，支持高效的数组操作、数学函数、线性代数、随机数生成等。
使用场景： NumPy 适用于处理大规模的数值数据，进行数组运算和数学操作。
如果你主要关注于数组的创建、切片、索引以及基本的数学运算，NumPy 是一个强大而高效的工具。
SciPy：

功能： SciPy 在 NumPy 的基础上构建，提供了更多高级的科学计算功能，
包括优化、插值、统计、信号处理、图像处理、常微分方程求解等。
它是为解决更复杂的科学和工程问题而设计的。
使用场景： 如果你需要进行更高级的科学计算，例如优化问题、
信号处理、统计分析等，那么SciPy 将是你的首选。SciPy
还提供了丰富的库和函数，用于解决各种科学问题。
总体而言，这两个库通常一起使用。NumPy 提供了基础的数组操作，
而 SciPy 构建在其基础之上，提供了更多领域特定的功能。
在进行科学计算和数据分析时，你可能会同时使用它们，充分发挥它们的优势。

因此，选择使用哪个库取决于你的具体需求。
如果主要进行数组操作和基本数学计算，NumPy 是一个强大的选择。
如果你需要更广泛的科学计算功能，包括优化、统计分析、信号处理等，那么SciPy 将提供更多的工具。
"""
# 以下是 Scipy 的一些主要子模块及其功能：

# （1）scipy.optimize： 优化--提供了用于寻找函数最小值或根的优化算法，包括全局优化、约束优化等。
import numpy as np
from scipy.optimize import minimize


# 定义目标函数
def objective(x):
    return x ** 2 + 4 * x + 4


# 使用优化算法寻找最小值
result = minimize(objective, 0)
print(result)

# （2）scipy.signal： 信号处理--提供了信号处理所需的工具，包括滤波、频谱分析、傅里叶变换等。
from scipy import signal
import matplotlib.pyplot as plt

# 生成信号
t = signal.linspace(0, 1, 1000, endpoint=False)
signal_waveform = signal.square(2 * np.pi * 5 * t)

# 绘制信号
plt.plot(t, signal_waveform)
plt.show()

# （3）scipy.stats： 统计学--包含了统计学函数，用于描述和分析数据的统计性质，如概率分布、假设检验等。
from scipy.stats import norm

# 创建正态分布
dist = norm(loc=0, scale=1)

# 计算概率密度函数值
pdf_value = dist.pdf(0.5)
print(pdf_value)

# （4）scipy.interpolate： 插值(没有拉格朗日插值)--提供了一系列插值函数，用于根据有限的数据点估算未知的数据点。
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 创建插值函数
x = np.linspace(0, 10, 10)
y = np.sin(x)
f = interp1d(x, y, kind='linear')

# 绘制原始数据和插值结果
x_new = np.linspace(0, 10, 100)
plt.plot(x, y, 'o', label='原始数据')
plt.plot(x_new, f(x_new), '-', label='线性插值')
plt.legend()
plt.show()

# （5）scipy.linalg： 线性代数--提供了线性代数操作的功能，如矩阵求逆、特征值分解等。
from scipy.linalg import inv, det

# 创建矩阵
A = np.array([[1, 2], [3, 4]])

# 求逆和行列式
A_inv = inv(A)
A_det = det(A)

print(A_inv)
print(A_det)
