import pandas as pd
import matplotlib.pyplot as plt

# 使用中文注释
plt.rcParams['font.sans-serif'] = ['SimHei']

# 参数设置
matrix_width = 38080
matrix_height = 37800

# 读取Excel文件
df = pd.read_excel('附件2.xlsx')

# 提取数据
widths = df['宽度']
heights = df['高度']

# 计算左下角坐标
left_bottom_coords = []
x, y = 0, 0
max_y = 0
for (w, h) in zip(widths, heights):
    if x + w > matrix_width:
        x = 0
        y = max_y
    if y + h > matrix_height:
        print("矩阵空间不足")
        break
    left_bottom_coords.append((x, y))
    x += w
    max_y = max(max_y, y + h)

# 更新原文件中的【左下角X坐标】和【左下角Y坐标】
df['左下角X坐标'] = [lx for lx, ly in left_bottom_coords]
df['左下角Y坐标'] = [ly for lx, ly in left_bottom_coords]

# 保存更新后的Excel文件
df.to_excel('更新原文件坐标.xlsx', index=False)

# 绘制图像
fig, ax = plt.subplots(figsize=(12, 12))
for (lx, ly), (w, h) in zip(left_bottom_coords, zip(widths, heights)):
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
