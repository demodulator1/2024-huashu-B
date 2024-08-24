import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from scipy.spatial import distance_matrix
from matplotlib.colors import Normalize
from matplotlib import cm

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
dist_matrix = manhattan_distance_matrix(np.array(selected_points))

# 构建MST
mst = build_mst(dist_matrix)

# 计算MST的总权重
mst_weight = sum(edge_data['weight'] for u, v, edge_data in mst.edges(data=True))

# 生成图表
x_coords, y_coords = zip(*selected_points)
mst_edges = list(mst.edges())

# 创建网格
matrix_width = 38080
matrix_height = 37800
grid_width = 595
grid_height = 630

# 创建网格密度字典
grid_density = {}
for i in range(0, matrix_width, grid_width):
    for j in range(0, matrix_height, grid_height):
        grid_density[(i, j)] = 0

# 计算每条MST边所在网格的布线密度
for (u, v) in mst_edges:
    x1, y1 = selected_points[u]
    x2, y2 = selected_points[v]

    # 计算边所在的网格
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    x_grid_start = (x_min // grid_width) * grid_width
    y_grid_start = (y_min // grid_height) * grid_height

    x_grid_end = (x_max // grid_width) * grid_width
    y_grid_end = (y_max // grid_height) * grid_height

    for x in range(x_grid_start, x_grid_end + grid_width, grid_width):
        for y in range(y_grid_start, y_grid_end + grid_height, grid_height):
            grid_density[(x, y)] += 1  # 增加布线密度（路径长度）

# 计算每个网格的面积
grid_area = grid_width * grid_height

# 计算布线密度（路径长度除以网格面积）
grid_density = {k: v / grid_area for k, v in grid_density.items()}

# 创建热度图
heatmap = np.zeros((matrix_height // grid_height, matrix_width // grid_width))

for (x, y), density in grid_density.items():
    x_idx = x // grid_width
    y_idx = y // grid_height
    heatmap[y_idx, x_idx] = density

# 绘制热度图
plt.figure(figsize=(12, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest', norm=Normalize(vmin=0, vmax=np.max(heatmap)))
plt.colorbar(label='布线密度')
plt.title('布线密度热度图')
plt.xlabel('网格X轴索引')
plt.ylabel('网格Y轴索引')
plt.show()

from scipy.spatial import distance_matrix
from matplotlib.colors import Normalize
from matplotlib import cm

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
dist_matrix = manhattan_distance_matrix(np.array(selected_points))

# 构建MST
mst = build_mst(dist_matrix)

# 计算MST的总权重
mst_weight = sum(edge_data['weight'] for u, v, edge_data in mst.edges(data=True))

# 生成图表
x_coords, y_coords = zip(*selected_points)
mst_edges = list(mst.edges())

# 创建网格
matrix_width = 38080
matrix_height = 37800
grid_width = 595
grid_height = 630

# 创建网格密度字典
grid_density = {}
for i in range(0, matrix_width, grid_width):
    for j in range(0, matrix_height, grid_height):
        grid_density[(i, j)] = 0

# 计算每条MST边所在网格的布线密度
for (u, v) in mst_edges:
    x1, y1 = selected_points[u]
    x2, y2 = selected_points[v]

    # 计算边所在的网格
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    x_grid_start = (x_min // grid_width) * grid_width
    y_grid_start = (y_min // grid_height) * grid_height

    x_grid_end = (x_max // grid_width) * grid_width
    y_grid_end = (y_max // grid_height) * grid_height

    for x in range(x_grid_start, x_grid_end + grid_width, grid_width):
        for y in range(y_grid_start, y_grid_end + grid_height, grid_height):
            grid_density[(x, y)] += 1  # 增加布线密度（路径长度）

# 计算每个网格的面积
grid_area = grid_width * grid_height

# 计算布线密度（路径长度除以网格面积）
grid_density = {k: v / grid_area for k, v in grid_density.items()}

# 创建热度图
heatmap = np.zeros((matrix_height // grid_height, matrix_width // grid_width))

for (x, y), density in grid_density.items():
    x_idx = x // grid_width
    y_idx = y // grid_height
    heatmap[y_idx, x_idx] = density

# 绘制热度图
plt.figure(figsize=(12, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest', norm=Normalize(vmin=0, vmax=np.max(heatmap)))
plt.colorbar(label='布线密度')
plt.title('布线密度热度图')
plt.xlabel('网格X轴索引')
plt.ylabel('网格Y轴索引')
plt.show()
