import numpy as np
import matplotlib.pyplot as plt

# 给定坐标点
coords = [
    (28994,23885),
    (26971,24240),
    (26971,24321),
    (25809,24078),
    (25878,24474)
]

# 获取坐标的最小和最大值
x_coords, y_coords = zip(*coords)
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# 计算点到其他点曼哈顿距离的和，避免重复路径
def unique_manhattan_distance_sum(x, y, points):
    distances = []
    for px, py in points:
        distance = abs(x - px) + abs(y - py)
        if distance not in distances:
            distances.append(distance)
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

# 生成图表
x_coords, y_coords = zip(*coords)
best_x, best_y = best_point

plt.figure(figsize=(16, 8))
plt.scatter(x_coords, y_coords, color='blue', label='Given Points')
plt.scatter(best_x, best_y, color='red', label='Best Point', zorder=5)

# 绘制从每个给定点到最佳点的曼哈顿路径
for (x, y) in coords:
    plt.plot([x, x], [y, best_y], color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.plot([x, best_x], [best_y, best_y], color='green', linestyle='--', linewidth=0.8, alpha=0.7)

plt.title('Scatter Plot of Coordinates with Optimal Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # 保持比例一致
plt.show()
