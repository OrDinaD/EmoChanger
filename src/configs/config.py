import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='w', encoding='utf-8')
    ]
)

# Получаем логгер
logger = logging.getLogger("config")

# Пути к директориям
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

logger.debug(f"ROOT_DIR: {ROOT_DIR}")
logger.debug(f"DATA_DIR: {DATA_DIR}")
logger.debug(f"MODELS_DIR: {MODELS_DIR}")
logger.debug(f"LOGS_DIR: {LOGS_DIR}")
logger.debug(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
logger.debug(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")

# Параметры логирования
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Параметры аудио
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_AUDIO_LENGTH = 10  # максимальная длина аудио в секундах (уменьшил до 10 для быстроты)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

logger.debug(f"SAMPLE_RATE: {SAMPLE_RATE}")
logger.debug(f"MAX_AUDIO_LENGTH: {MAX_AUDIO_LENGTH}")
logger.debug(f"N_FFT: {N_FFT}, HOP_LENGTH: {HOP_LENGTH}, N_MELS: {N_MELS}")

# Параметры обучения
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

logger.debug(f"BATCH_SIZE: {BATCH_SIZE}")
logger.debug(f"VALIDATION_SPLIT: {VALIDATION_SPLIT}")
logger.debug(f"NUM_EPOCHS: {NUM_EPOCHS}")
logger.debug(f"LEARNING_RATE: {LEARNING_RATE}")

# Параметры модели
NUM_EMOTIONS = 5
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear"]
EMOTIONS = {
    "Нейтральный": 0,
    "Радостный": 1,
    "Грустный": 2,
    "Злой": 3,
    "Испуганный": 4
}

logger.debug(f"NUM_EMOTIONS: {NUM_EMOTIONS}")
logger.debug(f"EMOTION_LABELS: {EMOTION_LABELS}")
logger.debug(f"EMOTIONS: {EMOTIONS}")

# Параметры интерфейса
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SHOW_PROGRESS = True
AUTO_SAVE = True
AUTO_SAVE_INTERVAL = 300  # в секундах

logger.debug(f"WINDOW_WIDTH: {WINDOW_WIDTH}, WINDOW_HEIGHT: {WINDOW_HEIGHT}")
logger.debug(f"SHOW_PROGRESS: {SHOW_PROGRESS}")
logger.debug(f"AUTO_SAVE: {AUTO_SAVE}, AUTO_SAVE_INTERVAL: {AUTO_SAVE_INTERVAL}")

# Создаем необходимые директории
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info("Все необходимые директории созданы")
except Exception as e:
    logger.error(f"Ошибка при создании директорий: {str(e)}")

logger.info("Конфигурация загружена успешно") 