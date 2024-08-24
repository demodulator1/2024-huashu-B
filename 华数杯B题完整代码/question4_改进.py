import pandas as pd
import matplotlib.pyplot as plt

# 使用中文注释
plt.rcParams['font.sans-serif'] = ['SimHei']

# 参数设置
matrix_width = 38080
matrix_height = 37800
padding = 550  # 矩形之间的间距

# 读取Excel文件
df = pd.read_excel('附件2.xlsx')

# 提取数据
widths = df['宽度']
heights = df['高度']

# 计算排列矩阵的行列数
num_rectangles = len(widths)
rows = int((matrix_height + padding) / (max(heights) + padding))
cols = int((matrix_width + padding) / (max(widths) + padding))

# 更新矩形的左下角坐标
left_bottom_coords = []
x, y = 0, 0
for i in range(num_rectangles):
    w, h = widths[i], heights[i]
    if x + w > matrix_width:
        x = 0
        y += max(heights) + padding
    if y + h > matrix_height:
        print("矩阵空间不足")
        break
    left_bottom_coords.append((x, y))
    x += w + padding

# 检查数据长度是否匹配
if len(left_bottom_coords) != len(df):
    print(f"警告: 矩形数量 ({len(left_bottom_coords)}) 与数据行数 ({len(df)}) 不匹配")
    # 可能需要填充或截断
    left_bottom_coords = left_bottom_coords + [(None, None)] * (len(df) - len(left_bottom_coords))

# 更新原文件中的【左下角X坐标】和【左下角Y坐标】
df['左下角X坐标'] = [lx for lx, ly in left_bottom_coords]
df['左下角Y坐标'] = [ly for lx, ly in left_bottom_coords]

# 保存更新后的Excel文件
df.to_excel('更新原文件坐标.xlsx', index=False)

# 绘制图像
fig, ax = plt.subplots(figsize=(12, 12))
for (lx, ly), (w, h) in zip(left_bottom_coords, zip(widths, heights)):
    if lx is not None and ly is not None:
        rect = plt.Rectangle((lx, ly), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

# 设置坐标轴范围
ax.set_xlim(0, matrix_width)
ax.set_ylim(0, matrix_height)

# 确保坐标轴比例相同
ax.set_aspect('equal', 'box')

plt.title('电路单元重新排布图')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.show()
