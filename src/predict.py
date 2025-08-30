import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

import config


def predict_batch(model, inputs):
    '''
    :param model: 预测模型
    :param inputs: 输入上联
    :return: 输出下联
    '''
    # 5. 预测逻辑
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_length=40, num_beams=10)
    return output


def predict(text, model, device, tokenizer):
    '''
    :param text: 输入文本
    :param model: 预测模型
    :param device: 设备
    :param tokenizer: 分词器
    :return: 预测文本
    '''

    # 4. 处理输入
    inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False).to(device)

    # 5. 预测下联
    batch_result = predict_batch(model, inputs)
    output = tokenizer.decode(batch_result[0], skip_special_tokens=True)
    output = ''.join([i for i in output if i != ' '])

    return output


def run_predict():
    # 1. 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2.加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bart-base-chinese')
    print('分词器加载成功')

    # 3. 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODELS_DIR).to(device)
    print('模型加载成功')

    # 4. 输入界面
    print('欢迎使用对联生成模型（输入q或quit退出）')
    while True:
        text = input('> ')
        if text in ['q', 'quit']:
            break
        if text.strip() == '':
            print('输入不能为空！')
            continue

        result = predict(text, model, device, tokenizer)
        print(result)


if __name__ == '__main__':
    run_predict()
