import matplotlib.pyplot as plt
import random
import math
from copy import copy


# 寻找新的中心点的函数
def new(group):
    minimum = 10000
    o = []
    for x1 in range(min(group['x']), max(group['x'])):
        for y1 in range(min(group['y']), max(group['y'])):
            j = 0
            red_sum = 0
            while j < len(group['x']):
                red_sum += math.sqrt((group['x'][j] - x1) ** 2 + (group['y'][j] - y1) ** 2)
                j += 1
            o.append(red_sum)
            if red_sum < minimum:
                minimum = copy(red_sum)
                x2 = copy(x1)
                y2 = copy(y1)
    return x2, y2


# 根据中心点聚类并且着色的函数
def color(a, b, x, y):
    i = 0
    red = {'x': [], 'y': []}
    blue = {'x': [], 'y': []}
    black = {'x': [], 'y': []}
    while i < len(a):
        distance0 = math.sqrt((float(a[i]) - x[0]) ** 2 + (float(b[i]) - y[0]) ** 2)
        distance1 = math.sqrt((float(a[i]) - x[1]) ** 2 + (float(b[i]) - y[1]) ** 2)
        distance2 = math.sqrt((float(a[i]) - x[2]) ** 2 + (float(b[i]) - y[2]) ** 2)
        if min(distance0, distance1, distance2) == distance0:
            plt.plot(float(a[i]), float(b[i]), 'ro', label='red')
            red['x'].append(float(a[i]))
            red['y'].append(float(b[i]))
        elif min(distance0, distance1, distance2) == distance1:
            plt.plot(float(a[i]), float(b[i]), 'ro', label='blue')
            blue['x'].append(float(a[i]))
            blue['y'].append(float(b[i]))
        else:
            plt.plot(float(a[i]), float(b[i]), 'ro', label='black')
            black['x'].append(float(a[i]))
            black['y'].append(float(b[i]))
        i += 1
    return red, blue, black


def main():
    # 读取数据
    file = open('./data/data.txt')
    a, b = [], []
    for line in file.readlines():
        data = line.strip().split(',')
        a.append(data[0])
        b.append(data[1])
    file.close()

    # 随机选取3个初始中心点
    x = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
    y = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
    red, blue, black = color(a, b, x, y)
    plt.plot(x[0], y[0], 'x', color='red', markersize=15, label='red center')
    plt.plot(x[1], y[1], 'x', color='blue', markersize=15, label='blue center')
    plt.plot(x[2], y[2], 'x', color='black', markersize=15, label='black center')
    plt.legend()
    plt.axis([0, 25, 0, 25])
    plt.show()

    # 循环执行函数，直到收敛
    while (x[0], y[0]) != new(red) or (x[1], y[1]) != new(blue) or (x[2], y[2]) != new(black):
        x[0], y[0] = new(red)
        x[1], y[1] = new(blue)
        x[2], y[2] = new(black)
        red, blue, black = color(a, b, x, y)
        plt.plot(x[0], y[0], 'x', color='red', markersize=15, label='red center')
        plt.plot(x[1], y[1], 'x', color='blue', markersize=15, label='blue center')
        plt.plot(x[2], y[2], 'x', color='black', markersize=15, label='black center')
        plt.legend()
        plt.axis([0, 25, 0, 25])
        plt.show()


if __name__ == '__main__':
    main()
