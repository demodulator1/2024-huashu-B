import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix

# 读取Excel文件
input_file = '接口偏置坐标更新.xlsx'

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

# 计算MST的边在每个矩形内的路径长度
def is_point_in_rectangle(point, rect):
    x, y = point
    rect_x, rect_y, width, height = rect
    return rect_x <= x <= rect_x + width and rect_y <= y <= rect_y + height

def calculate_mst_length_in_rectangle(rect, mst_edges, points):
    rect_x, rect_y, width, height = rect
    total_length = 0
    for (u, v) in mst_edges:
        x1, y1 = points[u]
        x2, y2 = points[v]
        # 检查边是否完全在矩形内
        if all(is_point_in_rectangle(p, rect) for p in [(x1, y1), (x2, y2)]):
            total_length += abs(x2 - x1) + abs(y2 - y1)  # 曼哈顿距离
    return total_length

# 计算总长度除以矩形面积的比值并求和
total_ratio = 0
for _, row in df.iterrows():
    rect = (row['左下角X坐标'], row['左下角Y坐标'], row['宽度'], row['高度'])
    area = row['宽度'] * row['高度']
    mst_length_in_rect = calculate_mst_length_in_rectangle(rect, list(mst.edges()), selected_points)
    total_ratio += mst_length_in_rect / area

# 输出结果，保留更多小数位数
print(f"所有矩形的MST路径长度与矩形面积的比值之和是: {total_ratio:.10f}")
