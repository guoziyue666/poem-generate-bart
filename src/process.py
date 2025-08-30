import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import random
import warnings

warnings.filterwarnings("ignore")
from datasets import load_dataset
from transformers import AutoTokenizer
import csv
import config


def process():
    '''数据预处理'''
    print('开始处理数据')
    # 0. 读取文件内容
    with open(config.RAW_DATA_DIR / 'in.txt', 'r', encoding='utf-8') as f1:
        lines1 = [line.strip().replace(' ', '') for line in f1.readlines()]

    with open(config.RAW_DATA_DIR / 'out.txt', 'r', encoding='utf-8') as f2:
        lines2 = [line.strip().replace(' ', '') for line in f2.readlines()]

    # 准备CSV数据
    csv_data = [['in', 'out']]
    for i in range(len(lines1)):
        csv_data.append([lines1[i], lines2[i]])

    # 写入CSV文件
    with open(config.PROCESSED_DATA_DIR / 'processed.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    print("CSV文件已生成：processed.csv")

    # 1. 读取文件
    dataset = load_dataset('csv', data_files=str(config.PROCESSED_DATA_DIR / 'processed.csv'))['train']

    # 随机选择10%的数据
    dataset_size = len(dataset)
    sample_size = int(dataset_size * 0.1)
    random_indices = random.sample(range(dataset_size), sample_size)
    dataset = dataset.select(random_indices)

    # 2. 划分数据集
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # 3. 创建分词器并进行分词
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bart-base-chinese'))

    # 4. 定义分词函数
    def tokenize_function(examples):
        inputs = tokenizer(text=examples['in'],
                           text_target=examples['out'],
                           max_length=config.SEQ_LEN,
                           truncation=True,
                           padding='max_length',
                           return_token_type_ids=False)
        return inputs

    # 5. 应用分词函数
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['in', 'out']
    )

    # 6. 创建训练集和测试集
    tokenized_datasets.save_to_disk(config.PROCESSED_DATA_DIR)

    print('数据处理完成')


if __name__ == '__main__':
    process()
