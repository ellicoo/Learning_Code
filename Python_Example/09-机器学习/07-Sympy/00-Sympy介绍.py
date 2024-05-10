"""
SymPy（Symbolic Python） 是一个用于符号数学计算的Python库，它提供了符号计算的功能，使得用户能够执行代数运算、
求解方程、微积分、线性代数等符号计算任务。与数值计算库（如NumPy）不同，SymPy 处理的是符号表达式，而不是数值。
"""
# 以下是 SymPy 的主要特点和用法：
# （1）符号表达式： SymPy 允许用户创建符号变量，这些变量可以用于构建符号表达式。这使得符号计算可以处理未知数、符号常数等。
from sympy import symbols, Eq, solve

# 创建符号变量
x, y = symbols('x y')

# 创建符号表达式
expr = x + 2*y

#（2）方程求解： SymPy 可以用于求解代数方程。用户可以定义方程，然后使用 solve 函数找到方程的根。
# 定义方程
equation = Eq(x**2 + y, 0)

# 解方程
solutions = solve(equation, x)

#（3）微积分： SymPy 提供了符号微积分的功能，包括求导、积分等。
from sympy import diff, integrate

# 求导
derivative = diff(x**2 + 3*x, x)

# 积分
integral = integrate(x**2 + 3*x, x)

#（4）线性代数： SymPy 支持符号线性代数运算，包括矩阵操作、行简化、解线性方程组等。
from sympy import Matrix, solve_linear_system

# 创建矩阵
A = Matrix([[1, 2], [3, 4]])

# 行简化
reduced_form = A.rref()

# 解线性方程组
linear_system = Eq(2*x + y, 3)
solutions = solve_linear_system([linear_system, equation])

#（5）绘图： SymPy 与 Matplotlib 集成，可以用于绘制符号表达式的图形。）
from sympy.plotting import plot

# 绘制符号表达式的图形
plot(x**2, (x, -5, 5))
