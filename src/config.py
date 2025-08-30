from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据路径
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
PRE_TRAINED_DIR = ROOT_DIR / 'pre_trained'
TRAIN_NAME = 'train.jsonl'
TEST_NAME = 'test.jsonl'

# 模型和日志路径
MODELS_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# 训练参数
SEQ_LEN = 32
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4