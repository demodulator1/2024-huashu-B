import numpy as np

# 给定坐标点
coords = [
    (28994, 23885),
    (26971, 24240),
    (26971, 24321),
    (25809, 24078),
    (25878, 24474)
]

# 获取坐标的最小和最大值
x_coords, y_coords = zip(*coords)
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# 去除重复路径的距离计算
def unique_manhattan_distance_sum(x, y, points):
    distances = set()
    for px, py in points:
        distances.add((abs(x - px) + abs(y - py)))
    return sum(distances)

# 初始化最小距离和最优点
min_distance = float('inf')
best_point = None

# 遍历矩形中的所有点
for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        distance_sum = unique_manhattan_distance_sum(x, y, coords)
        if distance_sum < min_distance:
            min_distance = distance_sum
            best_point = (x, y)

# 输出结果
print(f"使曼哈顿距离之和最小的点是: {best_point}")
print(f"该点到所有坐标的曼哈顿距离之和是: {min_distance}")
