import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#使用中文注释
plt.rcParams['font.sans-serif']=['SimHei']

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


# 创建图形
fig, ax = plt.subplots()

# 绘制矩形和接口点
for _, row in df.iterrows():
    x = row['左下角X坐标']
    y = row['左下角Y坐标']
    width = row['宽度']
    height = row['高度']

    # 绘制矩形
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # 绘制接口点
    intf_coords = parse_coordinates(row['调整后的接口偏置坐标'])
    for (px, py) in intf_coords:
        ax.plot(px, py, 'bo')  # 'bo' 表示蓝色圆点

# 设置坐标轴范围
ax.set_xlim(df['左下角X坐标'].min() - 10, df['左下角X坐标'].max() + df['宽度'].max() + 10)
ax.set_ylim(df['左下角Y坐标'].min() - 10, df['左下角Y坐标'].max() + df['高度'].max() + 10)

# 设置坐标轴标签
ax.set_xlabel('X坐标')
ax.set_ylabel('Y坐标')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('电路单元接口点坐标图')
plt.grid(True)
plt.show()
