import pandas as pd
import matplotlib.pyplot as plt

#使用中文注释
plt.rcParams['font.sans-serif']=['SimHei']

# 导入 Excel 数据
df = pd.read_excel('附件2.xlsx')

# 绘制矩形
plt.figure(figsize=(12, 8))

for index, row in df.iterrows():
    x = row['左下角X坐标']
    y = row['左下角Y坐标']
    width = row['宽度']
    height = row['高度']

    # 矩形的四个顶点
    rectangle = plt.Rectangle((x, y), width, height, fill=None, edgecolor='r', linewidth=2)

    plt.gca().add_patch(rectangle)

# 设置坐标轴范围
plt.xlim(df['左下角X坐标'].min() - 10, df['左下角X坐标'].max() + df['宽度'].max() + 10)
plt.ylim(df['左下角Y坐标'].min() - 10, df['左下角Y坐标'].max() + df['高度'].max() + 10)

# 添加网格和标签
plt.grid(True)
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.title('电路单元初始排布情况')

plt.gca().set_aspect('equal', adjustable='box')  # 保持比例一致
plt.show()
