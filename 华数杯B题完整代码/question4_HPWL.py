import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from scipy.spatial import distance_matrix

# 使用中文注释
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取Excel文件
input_file = '接口偏置坐标更新2.xlsx'

# 读取Excel中的数据
df = pd.read_excel(input_file)

# 解析接口偏置坐标
def parse_coordinates(coord_str):
    coord_str = coord_str.strip()
    coords = coord_str.split('), (')
    coords[0] = coords[0].strip('(')
    coords[-1] = coords[-1].strip(')')
    return [tuple(map(int, coord.split(','))) for coord in coords]

# 从每个单元格中选择一个坐标点
selected_points = []
for _, row in df.iterrows():
    intf_coords = parse_coordinates(row['调整后的接口偏置坐标'])
    if intf_coords:  # 确保列表不为空
        selected_points.append(intf_coords[0])  # 这里只选择第一个坐标点

# 去重
selected_points = list(set(selected_points))

# 计算HPWL距离矩阵
def hpwl_distance_matrix(points):
    num_points = len(points)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            hpwl = (abs(x2 - x1) + abs(y2 - y1)) / 2
            dist_matrix[i, j] = hpwl
            dist_matrix[j, i] = hpwl
    return dist_matrix

# 构建最小生成树（MST）
def build_mst(dist_matrix):
    G = nx.Graph()
    num_points = dist_matrix.shape[0]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    mst = nx.minimum_spanning_tree(G)
    return mst

# 计算HPWL距离矩阵
dist_matrix = hpwl_distance_matrix(np.array(selected_points))

# 构建MST
mst = build_mst(dist_matrix)

# 计算MST的总权重
mst_weight = sum(edge_data['weight'] for u, v, edge_data in mst.edges(data=True))

# 输出结果
print(f"最小生成树的总HPWL距离是: {mst_weight}")

# 生成图表
x_coords, y_coords = zip(*selected_points)
mst_edges = list(mst.edges())

plt.figure(figsize=(16, 8))
plt.scatter(x_coords, y_coords, color='blue', label='接口点')

# 仅沿x轴和y轴绘制连接线
for (u, v) in mst_edges:
    x1, y1 = selected_points[u]
    x2, y2 = selected_points[v]
    # 绘制水平和垂直线段
    plt.plot([x1, x1], [y1, y2], color='green', linestyle='--', linewidth=0.8, alpha=0.7)  # 垂直线
    plt.plot([x1, x2], [y2, y2], color='green', linestyle='--', linewidth=0.8, alpha=0.7)  # 水平线

# 绘制矩形
for _, row in df.iterrows():
    x = row['左下角X坐标']
    y = row['左下角Y坐标']
    width = row['宽度']
    height = row['高度']
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

plt.title('所有电路单元连接的HPWL距离')
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # 保持比例一致
plt.show()
