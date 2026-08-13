import pandas as pd
import re

# 读取原始Excel文件
input_file = r"D:\med_data\ai\translate\location_trans_size.xlsx"
output_file = r"D:\med_data\ai\translate\location_60000.xlsx"

# 读取Excel
df = pd.read_excel(input_file)

print(f"原始表格形状: {df.shape}")
print(f"原始列名: {df.columns.tolist()}")
print(f"前5行数据预览:\n{df.head()}\n")


# 提取Filename列，处理病历号
def process_filename(filename):
    """
    从文件名中提取病历号并加60000
    文件名格式: ANY_病历号_图像号
    例如: ANY_001_0 -> ANY_601_0
    """
    # 使用正则表达式匹配病历号
    # 模式: ANY_(\d+)_(\d+)
    match = re.search(r'ANY_(\d+)_(\d+)', str(filename))

    if match:
        case_num = int(match.group(1))  # 病历号
        image_num = int(match.group(2))  # 图像号

        # 病历号加60000
        new_case_num = case_num + 60000

        # 重新构造文件名
        new_filename = f"ANY_{new_case_num}_{image_num}"
        return new_filename
    else:
        # 如果格式不匹配，尝试只匹配数字部分
        numbers = re.findall(r'\d+', str(filename))
        if len(numbers) >= 2:
            case_num = int(numbers[0])
            image_num = int(numbers[1])
            new_case_num = case_num + 60000
            new_filename = f"ANY_{new_case_num}_{image_num}"
            return new_filename
        else:
            print(f"警告: 无法解析文件名 {filename}")
            return filename


# 创建新的DataFrame
# 第一列：处理后的文件名
df['Processed_Filename'] = df['Filename'].apply(process_filename)

# 重新排列列顺序：新文件名放在第一列，其他列按原顺序
# 获取除原Filename外的其他列
other_columns = [col for col in df.columns if col != 'Filename' and col != 'Processed_Filename']
# 新列顺序：Processed_Filename, 其他原列（不包括原Filename）
new_columns = ['Processed_Filename'] + other_columns

# 创建新的DataFrame
df_new = df[new_columns].copy()

# 重命名列名：将Processed_Filename改为Filename（或者保持原名）
df_new = df_new.rename(columns={'Processed_Filename': 'Filename'})

print(f"处理后表格形状: {df_new.shape}")
print(f"新列名: {df_new.columns.tolist()}")
print(f"前5行数据预览:\n{df_new.head()}")

# 保存到新Excel文件
df_new.to_excel(output_file, index=False)

print(f"\n成功保存到: {output_file}")


# 可选：显示一些统计信息（修复f-string中的反斜杠问题）
def extract_case_num(filename):
    """从文件名中提取病历号"""
    match = re.search(r'(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return 0


# 计算原始病历号范围
original_nums = df['Filename'].apply(extract_case_num)
min_original = original_nums.min()
max_original = original_nums.max()

# 计算新病历号范围
new_nums = df_new['Filename'].apply(extract_case_num)
min_new = new_nums.min()
max_new = new_nums.max()

print("\n处理统计:")
print(f"总行数: {len(df_new)}")
print(f"原始病历号范围: {min_original} - {max_original}")
print(f"新病历号范围: {min_new} - {max_new}")