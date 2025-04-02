import os

# Пути к директориям
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Параметры логирования
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Параметры аудио
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Параметры модели
NUM_EMOTIONS = 5
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear"]

# Параметры интерфейса
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Создаем необходимые директории
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True) 