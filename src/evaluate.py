import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.predict import predict_batch


def evaluate(model, test_dataloader, device):
    '''
    :param model: 预测模型
    :param test_dataloader: 测试集
    :param device: 设备
    :return: top1准确率，top5准确率
    '''
    total_count = 0
    acc = 0
    for inputs in tqdm(test_dataloader, desc="评估："):
        targets = inputs.pop('label').tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        batch_result = predict_batch(model, inputs)

        for result, target in zip(batch_result, targets):
            result = 1 if result > 0.5 else 0
            if target == result:
                acc += 1
            total_count += 1
    return acc / total_count


def run_evaluate():
    # 1. 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据集
    test_dataloader = get_dataloader(False)

    # 3. 创建模型
    model = ReviewAnalyseModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    # 4. 计算准确率
    acc = evaluate(model, test_dataloader, device)
    print('评估准确率：')
    print(f'acc: {acc:.4f}')


if __name__ == '__main__':
    run_evaluate()
