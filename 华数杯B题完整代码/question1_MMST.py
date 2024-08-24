import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import networkx as nx

#使用中文注释
plt.rcParams['font.sans-serif']=['SimHei']

coords = [
    (33800,15303),
    (31437,16378),
    (30938,16236),
    (30883,16694),
    (31039,15674)
]

# 计算曼哈顿距离矩阵
def manhattan_distance_matrix(points):
    return distance_matrix(points, points, p=1)

# 构建最小生成树（MST）
def build_mst(dist_matrix):
    G = nx.Graph()
    num_points = dist_matrix.shape[0]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    mst = nx.minimum_spanning_tree(G)
    return mst

# 计算曼哈顿距离矩阵
dist_matrix = manhattan_distance_matrix(np.array(coords))

# 构建MST
mst = build_mst(dist_matrix)

# 计算MST的总权重
mst_weight = sum(edge_data['weight'] for u, v, edge_data in mst.edges(data=True))

# 输出结果
print(f"最小生成树的总曼哈顿距离是: {mst_weight}")

# 生成图表
x_coords, y_coords = zip(*coords)
mst_edges = list(mst.edges())

plt.figure(figsize=(16, 8))
plt.scatter(x_coords, y_coords, color='blue', label='连接接口坐标')

# 仅沿x轴和y轴绘制连接线
for (u, v) in mst_edges:
    x1, y1 = coords[u]
    x2, y2 = coords[v]
    # 绘制水平和垂直线段
    plt.plot([x1, x1], [y1, y2], color='green', linestyle='--', linewidth=0.8, alpha=0.7)  # 垂直线
    plt.plot([x1, x2], [y2, y2], color='green', linestyle='--', linewidth=0.8, alpha=0.7)  # 水平线

plt.title('Group17坐标的最小生成树（曼哈顿距离）')
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # 保持比例一致
plt.show()
