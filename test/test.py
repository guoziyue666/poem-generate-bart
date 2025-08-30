import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from src import config
import csv

# 文件路径
file1_path = 'file1.txt'  # 请替换为实际文件路径
file2_path = 'file2.txt'  # 请替换为实际文件路径

# 读取文件内容
with open(config.RAW_DATA_DIR / 'in.txt', 'r', encoding='utf-8') as f1:
    lines1 = [line.strip().replace(' ', '') for line in f1.readlines()]

with open(config.RAW_DATA_DIR / 'out.txt', 'r', encoding='utf-8') as f2:
    lines2 = [line.strip().replace(' ', '') for line in f2.readlines()]

# 检查行数是否一致
if len(lines1) != len(lines2):
    print("错误：两个文件行数不一致")
else:
    print("正确：两个文件行数一致")

# 准备CSV数据
csv_data = [['in', 'out']]
for i in range(len(lines1)):
    csv_data.append([lines1[i], lines2[i]])

# 写入CSV文件（例如输出到output.csv）
with open(config.PROCESSED_DATA_DIR / 'processed.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)
print("CSV文件已生成：output.csv")
