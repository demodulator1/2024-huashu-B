import pandas as pd

# 读取Excel文件
input_file = '更新原文件坐标.xlsx'
output_file = '接口偏置坐标更新2.xlsx'

# 读取Excel中的数据
df = pd.read_excel(input_file)

# 确保有必要的列
assert '左下角X坐标' in df.columns
assert '左下角Y坐标' in df.columns
assert '接口偏置坐标' in df.columns


# 解析接口偏置坐标
def parse_coordinates(coord_str):
    # 去除括号并分割每个坐标对
    coord_str = coord_str.strip()
    coords = coord_str.split('), (')
    coords[0] = coords[0].strip('(')
    coords[-1] = coords[-1].strip(')')

    # 转换为整数元组
    return [tuple(map(int, coord.split(','))) for coord in coords]


# 应用偏置调整
def adjust_coordinates(row):
    x_offset = row['左下角X坐标']
    y_offset = row['左下角Y坐标']

    # 解析原始接口偏置坐标
    intf_coords = parse_coordinates(row['接口偏置坐标'])

    # 计算新的接口偏置坐标
    new_coords = [(x + x_offset, y + y_offset) for x, y in intf_coords]

    # 格式化为字符串
    return ', '.join([f"({x}, {y})" for x, y in new_coords])


# 应用调整
df['调整后的接口偏置坐标'] = df.apply(adjust_coordinates, axis=1)

# 写入新的Excel文件
df.to_excel(output_file, index=False)

print(f"调整后的数据已保存到 {output_file}")
