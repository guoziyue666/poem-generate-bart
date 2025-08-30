import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import warnings

warnings.filterwarnings("ignore")
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from src import config
from src.dataset import get_dataloader


def train_one_epoch(model, optimizer, train_dataloader, device):
    '''
    :param model: 模型
    :param optimizer: 优化器
    :param train_dataloader: 训练数据集
    :param device: 设备
    :return: 每轮平均损失
    '''
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc='Train: '):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(train_dataloader)


def train():
    # 1. 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据集
    train_dataloader = get_dataloader()

    # 3. 创建模型
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODELS_DIR).to(device)

    # 4. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. tensorboard可视化
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 6. 训练
    min_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print('=' * 10, f'EPOCH {epoch}', '=' * 10)
        avg_loss = train_one_epoch(model, optimizer, train_dataloader, device)
        writer.add_scalar('loss', avg_loss, epoch)
        print(f'Train Loss: {avg_loss}')

        if avg_loss < min_loss:
            min_loss = avg_loss
            model.save_pretrained(config.MODELS_DIR)
            print('保存模型成功')
    writer.close()


if __name__ == '__main__':
    train()
