"""
PuLP（Python Linear Programming Library）是一个用于线性规划（LP）的Python库，它提供了简单且灵活的接口来构建和求解线性规划问题。
线性规划是一种数学优化技术，用于在给定约束条件下最大化或最小化线性目标函数。

以下是 PuLP 的主要特点和用法：
"""
# （1）线性规划问题建模： PuLP 允许用户使用简单的语法和高级抽象来定义线性规划问题。通过声明变量、目标函数和约束，用户可以轻松地构建问题的数学模型。
from pulp import LpMaximize, LpProblem, LpVariable, value

# 创建线性规划问题
prob = LpProblem("Maximize_Profit", LpMaximize)

# 定义变量
x = LpVariable("x", lowBound=0)  # 非负变量
y = LpVariable("y", lowBound=0)

# 定义目标函数
prob += 3 * x + 2 * y, "Objective"

# 添加约束
prob += 2 * x + y <= 20
prob += 4 * x - 5 * y >= -10

#（2）求解器支持： PuLP 提供了对多个线性规划求解器的支持，包括 open-source 的 COIN-OR CBC 求解器和商业求解器如 CPLEX、Gurobi。用户可以根据自己的需求选择合适的求解器。
# 使用 CBC 求解器求解问题
prob.solve()

#（3）结果解释和输出： PuLP 允许用户获取最优解、目标函数值以及变量值等结果。此外，用户还可以将问题和解的详细信息输出为文本或其他格式。
# 获取结果
print("Status:", LpProblem.status[prob.status])
print("Optimal value:", value(prob.objective))
print("Optimal values for variables:")
for v in prob.variables():
    print(f"{v.name}: {v.varValue}")

#灵活性和可扩展性： PuLP 支持整数线性规划、混合整数线性规划等多种扩展形式，并提供了更多高级特性，如松弛变量、目标函数系数的敏感性分析等。

#PuLP 是一个轻量级且易于使用的线性规划库，适用于简单到中等规模的优化问题。对于更复杂的问题，可能需要考虑使用更专业的商业求解器。
# PuLP 的优势之一是它与 Python 的兼容性和简单易懂的API，使得用户能够快速上手和解决线性规划问题。