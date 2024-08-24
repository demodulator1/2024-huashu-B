import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
import pandas as pd

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

# 从Excel文件读取数据
df = pd.read_excel('附件1.xlsx')

# 确保数据列包含有效的坐标
data = df['对应连线接口坐标']
results = []

# 处理每一行的坐标
for i in range(len(data)):
    try:
        # 将字符串解析为坐标列表
        coords_str = data[i]
        coords = np.array(eval(coords_str))

        if coords.ndim == 1:
            coords = np.expand_dims(coords, axis=0)

        # 计算曼哈顿距离矩阵
        dist_matrix = manhattan_distance_matrix(coords)

        # 构建MST
        mst = build_mst(dist_matrix)

        # 计算MST的总权重
        mst_weight = sum(edge_data['weight'] for u, v, edge_data in mst.edges(data=True))

        # 添加结果到列表
        results.append(mst_weight)
    except Exception as e:
        results.append(f"处理第 {i} 行数据时出错: {e}")

# 将结果添加到DataFrame中
df['MST总曼哈顿距离'] = results

# 保存到Excel文件
df.to_excel('附件1_带结果.xlsx', index=False)
